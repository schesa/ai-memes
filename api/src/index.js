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
        template_id: 112126428, // distacted boyfriend meme
        username: process.env.IMGFLIP_USERNAME,
        password: process.env.IMGFLIP_PASSWORD,
        "boxes[0][text]": "Making memes using Ai",
        "boxes[1][text]": "9GAG Users",
        "boxes[2][text]": "Making memes",
      },
    }
  )
  .then((res) => {
    console.log(res.data.data);
  })
  .catch((error) => {
    console.error(error);
  });
