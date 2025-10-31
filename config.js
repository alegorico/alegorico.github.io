// Config for alegorico.github.io
var config = {

  // ID of element to attach CMS.js to
  elementId: 'cms',

  // Mode 'GITHUB' for Github Pages, 'SERVER' for Self Hosted
  mode: 'GITHUB',

  // Github configuration
  github: {
    username: 'alegorico',
    repo: 'alegorico.github.io',
    branch: 'gh-pages',
    host: 'https://api.github.com',
  },

  // The name of the layouts directory.
  layoutDirectory: 'layouts/pure-blog',

  // The error layout template name.
  errorLayout: 'error',

  // The URL that will be the default view that will initially load
  defaultView: 'posts',

  // These are the types of content to load.
  types: [
    {
      name: 'posts',
      layout: { list: 'post-list', single: 'post' },
    },
    {
      name: 'pages',
      layout: { list: 'page-list', single: 'page' },
    },
  ],
};

// Initialize CMS.js
var blog = CMS(config);
