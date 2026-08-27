import React, { useRef, useEffect, useState } from 'react';
import type { MatchPoint } from '../api/client';

interface MatchPointOverlayProps {
  sourceFile: File;
  referenceFile: File;
  matches: MatchPoint[];
}

export const MatchPointOverlay: React.FC<MatchPointOverlayProps> = ({
  sourceFile,
  referenceFile,
  matches
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  const [sourceImg, setSourceImg] = useState<HTMLImageElement | null>(null);
  const [referenceImg, setReferenceImg] = useState<HTMLImageElement | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);

  // Revoke old object URLs on change
  useEffect(() => {
    const srcUrl = URL.createObjectURL(sourceFile);
    const refUrl = URL.createObjectURL(referenceFile);

    const sImg = new Image();
    sImg.src = srcUrl;
    sImg.onload = () => setSourceImg(sImg);

    const rImg = new Image();
    rImg.src = refUrl;
    rImg.onload = () => setReferenceImg(rImg);

    return () => {
      URL.revokeObjectURL(srcUrl);
      URL.revokeObjectURL(refUrl);
    };
  }, [sourceFile, referenceFile]);

  // Redraw canvas loop
  useEffect(() => {
    if (!canvasRef.current || !sourceImg || !referenceImg) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const targetHeight = 400;
    const sScale = targetHeight / sourceImg.height;
    const sWidth = sourceImg.width * sScale;

    const rScale = targetHeight / referenceImg.height;
    const rWidth = referenceImg.width * rScale;

    // Adjust canvas size
    canvas.width = sWidth + rWidth + 20;
    canvas.height = targetHeight;

    // Draw dark background
    ctx.fillStyle = '#030406';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw source and reference images
    ctx.drawImage(sourceImg, 0, 0, sWidth, targetHeight);
    ctx.drawImage(referenceImg, sWidth + 20, 0, rWidth, targetHeight);

    // Draw match lines
    matches.forEach((m, idx) => {
      const sx = m.source_pt[0] * sScale;
      const sy = m.source_pt[1] * sScale;
      const rx = sWidth + 20 + m.reference_pt[0] * rScale;
      const ry = m.reference_pt[1] * rScale;

      const isHovered = idx === hoveredIndex;

      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(rx, ry);
      
      if (isHovered) {
        ctx.strokeStyle = '#39ff14';
        ctx.lineWidth = 3;
      } else {
        // Dim other lines when one is hovered
        ctx.strokeStyle = hoveredIndex !== null ? 'rgba(0, 240, 255, 0.05)' : 'rgba(0, 240, 255, 0.2)';
        ctx.lineWidth = 1;
      }
      ctx.stroke();

      // Source keypoint node
      ctx.beginPath();
      ctx.arc(sx, sy, isHovered ? 5 : 3, 0, 2 * Math.PI);
      ctx.fillStyle = isHovered ? '#39ff14' : '#d000ff';
      ctx.fill();

      // Reference keypoint node
      ctx.beginPath();
      ctx.arc(rx, ry, isHovered ? 5 : 3, 0, 2 * Math.PI);
      ctx.fillStyle = isHovered ? '#39ff14' : '#d000ff';
      ctx.fill();
    });
  }, [sourceImg, referenceImg, matches, hoveredIndex]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !sourceImg || !referenceImg) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const targetHeight = 400;
    const sScale = targetHeight / sourceImg.height;
    const sWidth = sourceImg.width * sScale;
    const rScale = targetHeight / referenceImg.height;

    let closestIndex: number | null = null;
    let minDist = 15; // Threshold in pixels

    matches.forEach((m, idx) => {
      const sx = m.source_pt[0] * sScale;
      const sy = m.source_pt[1] * sScale;
      const rx = sWidth + 20 + m.reference_pt[0] * rScale;
      const ry = m.reference_pt[1] * rScale;

      const distSrc = Math.hypot(x - sx, y - sy);
      const distRef = Math.hypot(x - rx, y - ry);
      
      const dx = rx - sx;
      const dy = ry - sy;
      const t = Math.max(0, Math.min(1, ((x - sx) * dx + (y - sy) * dy) / (dx * dx + dy * dy + 1e-8)));
      const px = sx + t * dx;
      const py = sy + t * dy;
      const distLine = Math.hypot(x - px, y - py);

      const d = Math.min(distSrc, distRef, distLine);
      if (d < minDist) {
        minDist = d;
        closestIndex = idx;
      }
    });

    if (closestIndex !== null) {
      setHoveredIndex(closestIndex);
      setTooltipPos({ x: x + 15, y: y + 15 });
    } else {
      setHoveredIndex(null);
      setTooltipPos(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredIndex(null);
    setTooltipPos(null);
  };

  const hoveredMatch = hoveredIndex !== null ? matches[hoveredIndex] : null;

  return (
    <div className="match-inspector">
      <div className="canvas-container" style={{ position: 'relative' }}>
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ cursor: hoveredIndex !== null ? 'pointer' : 'default' }}
        />
        {tooltipPos && hoveredMatch && (
          <div className="canvas-tooltip" style={{ left: tooltipPos.x, top: tooltipPos.y }}>
            <div className="canvas-tooltip-row">
              <span>Match ID:</span>
              <span>#{hoveredIndex}</span>
            </div>
            <div className="canvas-tooltip-row">
              <span>Source Pixel:</span>
              <span>({hoveredMatch.source_pt[0].toFixed(1)}, {hoveredMatch.source_pt[1].toFixed(1)})</span>
            </div>
            <div className="canvas-tooltip-row">
              <span>Reference Pixel:</span>
              <span>({hoveredMatch.reference_pt[0].toFixed(1)}, {hoveredMatch.reference_pt[1].toFixed(1)})</span>
            </div>
            <div className="canvas-tooltip-row">
              <span>Confidence (NCC):</span>
              <span>{hoveredMatch.confidence.toFixed(4)}</span>
            </div>
          </div>
        )}
      </div>
      <div className="inspector-info">
        <span>Total Inlier Tie-Points: <strong>{matches.length}</strong></span>
        <span>Hover over connections to inspect coordinates and correlation confidence values.</span>
      </div>
    </div>
  );
};
