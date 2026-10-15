# syntax=docker/dockerfile:1
FROM node:24-alpine AS build

ARG BACKEND_URL=http://backend:8000
ENV BACKEND_URL=${BACKEND_URL}

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM node:24-alpine AS run

ENV NODE_ENV=production
WORKDIR /web

COPY --from=build /web/.next/standalone ./
COPY --from=build /web/.next/static ./.next/static
COPY --from=build /web/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
