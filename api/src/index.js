const axios = require("axios");
require("dotenv").config();

// Image Captioning API basic usage
axios
  .post(
    "https://api.imgflip.com/caption_image",
    {},
    {
      params: {
        template_id: 14859329,
        username: process.env.IMGFLIP_USERNAME,
        password: process.env.IMGFLIP_PASSWORD,
        text0: "when the api",
        text1: "has parameters in post",
      },
    }
  )
  .then((res) => {
    console.log(res.data.data);
  })
  .catch((error) => {
    console.error(error);
  });

// Image Captioning API using boxes
axios
  .post(
    "https://api.imgflip.com/caption_image",
    {},
    {
      params: {
        // template_id: 102156234, // distacted boyfriend meme
        template_id: 405658, // grumpy cat meme
        username: process.env.IMGFLIP_USERNAME,
        password: process.env.IMGFLIP_PASSWORD,
        "boxes[0][text]": "when life gives you lemons",
        "boxes[1][text]": "throw them at people",
        // "boxes[2][text]": "Making memes",
        // to consider feature
        // boxes[1][force_caps]: 0,
      },
    }
  )
  .then((res) => {
    console.log(res.data.data);
  })
  .catch((error) => {
    console.error(error);
  });
