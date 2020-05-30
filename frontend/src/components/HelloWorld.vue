<template>
  <div class="hello">
    <div v-for="meme in allMemes" :key="meme.id">
      <h1>{{meme.caption.split('|').join(", ")}}</h1>
      <img :src="meme.url" :alt="meme.caption" />
    </div>
    <input type="text" v-model="memeCaption" />
    <input type="text" v-model="memeTemplate" />
    <button @click="onClicked">Add meme</button>
  </div>
</template>

<script>
import gql from "graphql-tag";

export default {
  name: "HelloWorld",
  methods: {
    async onClicked() {
      console.log(this.memeCaption);
      console.log(this.memeTemplate);
      const result = await this.$apollo.mutate({
        mutation: gql`
          mutation AddMeme($caption: String!, $templateid: String!) {
            addMeme(meme: { caption: $caption, templateid: $templateid }) {
              meme {
                id
              }
            }
          }
        `,
        // Parameters
        variables: {
          templateid: this.memeTemplate,
          caption: this.memeCaption
        }
      });
    }
  },
  data() {
    return { memeCaption: "", memeTemplate: "" };
  },
  apollo: {
    allMemes: gql`
      query allMemes {
        allMemes {
          id
          caption
          url
          created
          templateid
          templatename
        }
      }
    `
  },
  props: {
    // msg: String
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>

