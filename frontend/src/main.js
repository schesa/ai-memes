import Vue from 'vue'
import App from './App.vue'
import { createProvider } from './vue-apollo'
import VueMaterial from 'vue-material'
import 'vue-material/dist/vue-material.min.css'
import 'vue-material/dist/theme/default.css'
import VueClazyLoad from 'vue-clazy-load' // ES6 (Babel and others)

Vue.config.productionTip = false


new Vue({
  apolloProvider: createProvider(),
  render: h => h(App)
}).$mount('#app')

Vue.use(VueMaterial)
Vue.use(VueClazyLoad)