FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json vite.config.ts index.html ./
COPY web ./web
COPY server ./server
COPY shared ./shared
COPY configs ./configs
RUN npm run build
FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --from=build /app/server ./server
COPY --from=build /app/shared ./shared
COPY --from=build /app/configs ./configs
COPY package.json ./
COPY datas/test/*.csv ./datas/test/
EXPOSE 3001
CMD ["node", "node_modules/tsx/dist/cli.mjs", "server/index.ts"]
