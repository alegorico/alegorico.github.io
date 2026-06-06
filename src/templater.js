import { get } from './utils';
import { messages as msg, handleMessage } from './messages';

/**
 * Templating function that renders HTML templates.
 * @function
 * @param {string} text - HTML text to be evaluated.
 * @returns {string} Rendered template with injected data.
 */
export function Templater(text) {
  var code = 'var output="";';
  var index = 0;
  var regex = /<%=(.+?)%>|<%([\s\S]+?)%>/g;
  var match;
  while ((match = regex.exec(text)) !== null) {
    var html = text.slice(index, match.index);
    if (html) {
      code += 'output+=' + JSON.stringify(html) + ';';
    }
    if (match[1] !== undefined) {
      code += 'output+=(' + match[1] + ');';
    } else if (match[2] !== undefined) {
      code += match[2] + '\n';
    }
    index = regex.lastIndex;
  }
  var remaining = text.slice(index);
  if (remaining) {
    code += 'output+=' + JSON.stringify(remaining) + ';';
  }
  code += 'return output;';
  return new Function('data', code);
}

/**
 * Load template from URL.
 * @function
 * @async
 * @param {string} url - URL of template to load.
 * @param {object} data - Data to load into template.
 * @param {function} callback - Callback function
 */
export function loadTemplate(url, data, callback) {
  get(url, (success, error) => {
    if (error) {
      callback(success, error);
      return;
    }
    callback(Templater(success)(data), error);
  });
}

/**
 * Renders the layout into the main container.
 * @function renderLayout
 * @async
 * @param {string} layout - Filename of layout.
 * @param {object} data - Data passed to template.
 */
export function renderLayout(layout, config, data) {
  config.container.innerHTML = '';
  var url = [config.layoutDirectory, '/', layout, '.html'].join('');
  loadTemplate(url, data, (success, error) => {
    if (error) {
      handleMessage(msg['LAYOUT_LOAD_ERROR']);
    } else {
      config.container.innerHTML = success;
    }
  });
}
