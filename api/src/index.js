const axios = require("axios");
require("dotenv").config();

axios
  .post(
    "https://api.imgflip.com/caption_image",
    {},
    {
      params: {
        template_id: 14859329,
        username: process.env.IMGFLIP_USERNAME,
        password: process.env.IMGFLIP_PASSWORD,
        text0: "Cand ai o pitica",
        text1: "Dar nu te lasa in pace...."
      }
    }
  )
  .then(res => {
    console.log(res.data.data);
  })
  .catch(error => {
    console.error(error);
  });
