const axios = require("axios");
var grpc = require("grpc");
var protoLoader = require("@grpc/proto-loader");
require("dotenv").config();

var PROTO_PATH = __dirname + "/../memegenerator.proto";
var packageDefinition = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});
var memes_proto = grpc.loadPackageDefinition(packageDefinition).aimemes;

// // Image Captioning API basic usage
// axios
//   .post(
//     "https://api.imgflip.com/caption_image",
//     {},
//     {
//       params: {
//         template_id: 14859329,
//         username: process.env.IMGFLIP_USERNAME,
//         password: process.env.IMGFLIP_PASSWORD,
//         text0: "when the api",
//         text1: "has parameters in post",
//       },
//     }
//   )
//   .then((res) => {
//     console.log(res.data.data);
//   })
//   .catch((error) => {
//     console.error(error);
//   });

// // Image Captioning API using boxes
// axios
//   .post(
//     "https://api.imgflip.com/caption_image",
//     {},
//     {
//       params: {
//         // template_id: 102156234, // distacted boyfriend meme
//         template_id: 405658, // grumpy cat meme
//         username: process.env.IMGFLIP_USERNAME,
//         password: process.env.IMGFLIP_PASSWORD,
//         "boxes[0][text]": "when life gives you lemons",
//         "boxes[1][text]": "throw them at people",
//         // "boxes[2][text]": "Making memes",
//         // to consider feature
//         // boxes[1][force_caps]: 0,
//       },
//     }
//   )
//   .then((res) => {
//     console.log(res.data.data);
//   })
//   .catch((error) => {
//     console.error(error);
//   });

const postCaption = async (params) => {
  return axios.post(
    "https://api.imgflip.com/caption_image",
    {},
    {
      params,
    }
  );
  // .then((res) => {
  //   console.log(res.data.data);
  // })
  // .catch((error) => {
  //   console.error(error);
  // });
};

const getUrlForCaption = async (memeid, caption, callback) => {
  let url = "";
  let params = {
    // template_id: 102156234, // distacted boyfriend meme
    // template_id: 405658, // grumpy cat meme
    template_id: memeid,
    username: process.env.IMGFLIP_USERNAME,
    password: process.env.IMGFLIP_PASSWORD,
    "boxes[0][text]": "when life gives you lemons",
    "boxes[1][text]": "throw them at people",
    // "boxes[2][text]": "Making memes",
    // to consider feature
    // boxes[1][force_caps]: 0,
  };
  try {
    const boxes = caption.split("|");
    for (const [i, box] of boxes.entries()) {
      params[`boxes[${i}][text]`] = box;
    }
    console.log(params);

    const resp = await postCaption(params);
    console.log(resp.data);
    url = resp.data.data.url; // also has page with post
  } catch (err) {
    console.error(err);
  }
  callback(null, { url });
};

const getMemeUrl = (call, callback) => {
  const { memeid, caption } = call.request;
  getUrlForCaption(memeid, caption, callback);
  // callback(null, { url });
};

/**
 * Starts an RPC server that receives requests for the Greeter service at the
 * sample server port
 */
function main() {
  var server = new grpc.Server();
  server.addService(memes_proto.Memer.service, {
    getMemeUrl: getMemeUrl,
  });
  server.bind("0.0.0.0:50051", grpc.ServerCredentials.createInsecure());
  server.start();
}

main();
