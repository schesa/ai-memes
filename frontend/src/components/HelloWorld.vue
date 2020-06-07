<template>
  <div class="hello">
    <!-- <md-field>
      <label>Meme Caption</label>
      <md-input v-model="memeCaption"></md-input>
    </md-field>
    <md-field>
      <label>Meme Template</label>
      <md-input v-model="memeTemplate"></md-input>
    </md-field>
    <md-button class="md-raised  md-primary" @click="onClicked">Add meme</md-button>
    <input type="text" v-model="memeCaption" />
    <input type="text" v-model="memeTemplate" /> -->

    
    <div class="top md-layout md-gutter md-alignment-center-space-around">

      <form novalidate class="md-layout-item md-size-40" @submit.prevent="onClicked">
        <md-card>
            <md-card-header>
                <div class="md-title">Create your meme</div>
            </md-card-header>
        <md-card-content>
          <md-field>
            <label>Meme Caption</label>
            <md-input v-model="memeCaption"></md-input>
          </md-field>
          <md-field>
            <label>Template ID</label>
            <md-input v-model="memeTemplate"></md-input>
          </md-field>
        </md-card-content>
            <md-card-actions>
                <md-button type="submit" class="md-primary" :disabled="sending">Add</md-button>
            </md-card-actions>
        </md-card>
      </form>

      <div class="md-layout-item md-size-50">
            <md-button class="md-raised md-accent">Generate meme using AI</md-button>
            <div v-if="loading">
              <md-progress-spinner class="md-accent" md-mode="indeterminate" />
            </div>
      </div>
    </div>
<div class="md-inset">
</div>
    <div class="md-layout md-gutter md-alignment-top-space-around">
      <md-card class="meme md-layout-item md-size-15" v-for="meme in allMemes.filter(meme=>meme.caption!=='')" :key="meme.id">
        <md-card-media>
          <img :src="meme.url" :alt="meme.caption+meme.templateid" />
        </md-card-media>

        <md-card-content>
          {{meme.caption.split('|').join("\n")}}
        </md-card-content>
      </md-card>
    </div>
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
      this.loading = true;
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
      this.loading = false;
      console.log('result')
      console.log(result)
    }
  },
  data() {
    return { memeCaption: "", memeTemplate: "", loading: false };
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
.top{
  padding: 5em;
}
</style>

