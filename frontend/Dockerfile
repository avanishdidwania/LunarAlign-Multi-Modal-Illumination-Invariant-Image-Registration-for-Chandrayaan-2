# Stage 1: Build React production bundle
FROM node:20-alpine AS build

WORKDIR /app

# Copy package descriptors
COPY package*.json ./
RUN npm ci

# Copy full application code
COPY . .

# Build the production static distribution
RUN npm run build

# Stage 2: Serve using Nginx reverse proxy
FROM nginx:alpine

# Copy built distribution from Stage 1
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom Nginx routing and proxy configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
