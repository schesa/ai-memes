FROM node:14.3.0

WORKDIR /app
COPY package.json /app
RUN npm install
COPY . /app

RUN npm install

CMD node ./src/index.js

# grpc channel
EXPOSE 50051
