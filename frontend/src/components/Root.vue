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

    <div class="top md-layout md-gutter md-alignment-center-space-between">

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
      <div v-if="loading_general" class="md-layout-item md-size-10">
              <lottie-player src="https://assets9.lottiefiles.com/packages/lf20_gwBIWJ.json" mode="bounce" background="transparent"  speed="0.7"  style="width: 300px; height: 300px;"  loop  autoplay></lottie-player>            
      </div>

      <div class="md-layout-item md-size-40">
            <md-button @click="onGenerate" class="md-raised md-accent">Generate meme using AI</md-button>
            <!-- <div v-if="loading_general">
              <lottie-player src="https://assets9.lottiefiles.com/packages/lf20_gwBIWJ.json" mode="bounce" background="transparent"  speed="0.7"  style="width: 300px; height: 300px;"  loop  autoplay></lottie-player>            
            </div> -->
      </div>
    </div>
<div class="md-inset">
</div>
    <div class="md-layout md-gutter md-alignment-top-space-around">
      <md-card class="meme md-layout-item md-size-15" v-for="meme in allMemes.filter(meme=> (meme.caption != '' && meme.caption !== null) )" :key="meme.id">
        <md-card-media>
          <div v-if="loading_general">
            <!-- <md-progress-spinner class="md-primary" md-mode="indeterminate" /> -->
            <!-- <lottie-player src="https://assets10.lottiefiles.com/packages/lf20_7CAQeC.json"  background="transparent"  speed="0.7"  style="width: 300px; height: 300px;"  loop  autoplay></lottie-player> -->
            <lottie-player src="https://assets10.lottiefiles.com/packages/lf20_7CAQeC.json"  background="transparent"  speed="0.7"  style="width: 280px; height: 180px;"  loop  autoplay></lottie-player>
          </div>
          
          <div v-else>
            <clazy-load :src="meme.url">
              <img :src="meme.url">
              <div class="preloader" slot="placeholder">
                <md-progress-spinner class="md-primary" md-mode="indeterminate" />
              </div>
            </clazy-load>
            <!-- <img :src="meme.url" :alt="meme.caption+' '+meme.templateid" /> -->
          </div>
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
  name: "Root",
  methods: {
    async onClicked() {
      console.log(this.memeCaption);
      console.log(this.memeTemplate);
      this.loading_general = true;
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
      this.loading_general = false;
      console.log('result')
      console.log(result)
    }, 
    async onGenerate(){
        this.loading = true;
        this.loading_general = true;
        const result = await this.$apollo.mutate({
          mutation: gql`
            mutation AddMeme($templateid: String!) {
              addMeme(meme: { templateid: $templateid }) {
                meme {
                  id
                }
              }
            }
          `,
          // Parameters
          variables: {
            templateid: this.memeTemplate,
          }
        });
        console.log('result')
        console.log(result)
        // function sleep(ms) {
        //   return new Promise(resolve => setTimeout(resolve, ms));
        // }
        // await sleep(10000);
        // this.loading = false;
        // this.$apollo.queries.allMemes.refetch()

        var that = this;
        // setTimeout(function(){ 
        //   that.$apollo.queries.allMemes.refetch();
        //   that.loading = false;
        //  }, 10000);

        // repeat with the interval of 2 seconds
        // let timerId = setInterval(() => {
        //   // that.$apollo.queries.allMemes.refetch();
        //   // that.loading = !that.loading;
        // }, 1000);

        // after 5 seconds stop
        setTimeout(() => { that.$apollo.queries.allMemes.refetch(); that.loading=false; that.loading_general=false; }, 13000);
    }
  },
  data() {
    return { memeCaption: "", memeTemplate: "", loading: false, loading_general: false };
  },
  // created() {
  //   let ckeditor = document.createElement('script');   
  //   ckeditor.setAttribute('src',"https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js");
  //   document.head.appendChild(ckeditor);
  // },
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

