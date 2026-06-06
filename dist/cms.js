/*! @chrisdiana/cmsjs v2.0.1 | MIT (c) 2026 Chris Diana | https://github.com/chrisdiana/cms.js */
var CMS = (function () {
  'use strict';

  function _classCallCheck(a, n) {
    if (!(a instanceof n)) throw new TypeError("Cannot call a class as a function");
  }
  function _defineProperties(e, r) {
    for (var t = 0; t < r.length; t++) {
      var o = r[t];
      o.enumerable = o.enumerable || false, o.configurable = true, "value" in o && (o.writable = true), Object.defineProperty(e, _toPropertyKey(o.key), o);
    }
  }
  function _createClass(e, r, t) {
    return r && _defineProperties(e.prototype, r), Object.defineProperty(e, "prototype", {
      writable: false
    }), e;
  }
  function _toPrimitive(t, r) {
    if ("object" != typeof t || !t) return t;
    var e = t[Symbol.toPrimitive];
    if (void 0 !== e) {
      var i = e.call(t, r);
      if ("object" != typeof i) return i;
      throw new TypeError("@@toPrimitive must return a primitive value.");
    }
    return (String )(t);
  }
  function _toPropertyKey(t) {
    var i = _toPrimitive(t, "string");
    return "symbol" == typeof i ? i : i + "";
  }

  var defaults = {
    elementId: null,
    layoutDirectory: null,
    defaultView: null,
    errorLayout: null,
    mode: 'SERVER',
    github: null,
    types: [],
    plugins: [],
    frontMatterSeperator: /^---$/m,
    listAttributes: ['tags'],
    dateParser: /\d{4}-\d{2}(?:-\d{2})?/,
    dateFormat: function dateFormat(date) {
      return [date.getMonth() + 1, date.getDate(), date.getFullYear()].join('/');
    },
    extension: '.md',
    sort: undefined,
    markdownEngine: null,
    debug: false,
    messageClassName: 'cms-messages',
    onload: function onload() {},
    onroute: function onroute() {}
  };

  var messageContainer;
  var messages = {
    ELEMENT_ID_ERROR: 'ERROR: No element ID or ID incorrect. Check "elementId" parameter in config.',
    DIRECTORY_ERROR: 'ERROR: Error getting files. Make sure there is a directory for each type in config with files in it.',
    GET_FILE_ERROR: 'ERROR: Error getting the file',
    LAYOUT_LOAD_ERROR: 'ERROR: Error loading layout. Check the layout file to make sure it exists.',
    NOT_READY_WARNING: 'WARNING: Not ready to perform action'
  };

  /**
   * Creates message container element
   * @function
   * @param {string} classname - Container classname.
   */
  function createMessageContainer(classname) {
    messageContainer = document.createElement('div');
    messageContainer.className = classname;
    messageContainer.innerHTML = 'DEBUG';
    messageContainer.style.background = 'yellow';
    messageContainer.style.position = 'absolute';
    messageContainer.style.top = '0px';
    document.body.appendChild(messageContainer);
  }

  /**
   * Handle messages
   * @function
   * @param {string} message - Message.
   * @returns {string} message
   * @description
   * Used for debugging purposes.
   */
  function handleMessage(debug, message) {
    if (message === undefined) {
      message = debug;
      debug = true;
    }
    if (debug && messageContainer) {
      messageContainer.innerHTML = message;
    }
    return message;
  }

  /**
   * AJAX Get utility function.
   * @function
   * @async
   * @param {string} url - URL of the request.
   * @param {function} callback - Callback after request is complete.
   */
  function get(url, callback) {
    var req = new XMLHttpRequest();
    req.open('GET', url, true);
    req.onreadystatechange = function () {
      if (req.readyState === 4) {
        if (req.status === 200) {
          callback(req.response, false);
        } else {
          callback(req, req.statusText);
        }
      }
    };
    req.send();
  }

  /**
   * Extend utility function for extending objects.
   * @function
   * @param {object} target - Target object to extend.
   * @param {object} opts - Options to extend.
   * @param {function} callback - Callback function after completion.
   * @returns {object} Extended target object.
   */
  function extend(target, opts, callback) {
    var next;
    if (typeof opts === 'undefined') {
      opts = target;
    }
    for (next in opts) {
      if (Object.prototype.hasOwnProperty.call(opts, next)) {
        target[next] = opts[next];
      }
    }
    return target;
  }

  /**
   * Utility function for getting a function name.
   * @function
   * @param {function} func - The function to get the name
   * @returns {string} Name of function.
   */
  function getFunctionName(func) {
    if (func.name) return func.name;
    var ret = func.toString();
    ret = ret.substr('function '.length);
    ret = ret.substr(0, ret.indexOf('('));
    return ret;
  }

  /**
   * Checks if the file URL with file extension is a valid file to load.
   * @function
   * @param {string} fileUrl - File URL
   * @returns {boolean} Is valid.
   */
  function isValidFile(fileUrl, extension) {
    if (fileUrl) {
      var ext = fileUrl.split('.').pop();
      return ext === extension.replace('.', '') || ext === 'html' ? true : false;
    }
  }

  /**
   * Get URL paths without parameters.
   * @function
   * @returns {string} URL Path
   */
  function getPathsWithoutParameters() {
    return window.location.hash.split('/').map(function (path) {
      if (path.indexOf('?') >= 0) {
        path = path.substring(0, path.indexOf('?'));
      }
      return path;
    }).filter(function (path) {
      return path !== '#';
    });
  }

  /**
   * Get URL parameter by name.
   * @function
   * @param {string} name - Name of parameter.
   * @param {string} url - URL
   * @returns {string} Parameter value
   */
  function getParameterByName(name, url) {
    if (!url) url = window.location.href;
    name = name.replace(/[[]]/g, '\\$&');
    var regex = new RegExp('[?&]' + name + '(=([^&#]*)|&|#|$)'),
      results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, ' '));
  }

  /**
   * Get Github URL based on configuration.
   * @function
   * @param {string} type - Type of file.
   * @returns {string} GIthub URL
   */
  function getGithubUrl(type, gh) {
    var url = [gh.host, 'repos', gh.username, gh.repo, 'contents', type + '?ref=' + gh.branch];
    if (gh.prefix) url.splice(5, 0, gh.prefix);
    return url.join('/');
  }

  /**
   * Formats date string to datetime
   * @param {string} dateString - Date string to convert.
   * @returns {object} Formatted datetime
   */
  function getDatetime(dateStr) {
    var dt = new Date(dateStr);
    return new Date(dt.getTime() - dt.getTimezoneOffset() * -6e4);
  }

  /**
   * @param {string} filepath - Full file path including file name.
   * @returns {string} filename
   */
  function getFilenameFromPath(filepath) {
    return filepath.split('\\').pop().split('/').pop();
  }

  /**
   * Templating function that renders HTML templates.
   * @function
   * @param {string} text - HTML text to be evaluated.
   * @returns {string} Rendered template with injected data.
   */
  function Templater(text) {
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
  function loadTemplate(url, data, callback) {
    get(url, function (success, error) {
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
  function renderLayout(layout, config, data) {
    config.container.innerHTML = '';
    var url = [config.layoutDirectory, '/', layout, '.html'].join('');
    loadTemplate(url, data, function (success, error) {
      if (error) {
        handleMessage(messages['LAYOUT_LOAD_ERROR']);
      } else {
        config.container.innerHTML = success;
      }
    });
  }

  /**
   * marked v18.0.5 - a markdown parser
   * Copyright (c) 2018-2026, MarkedJS. (MIT License)
   * Copyright (c) 2011-2018, Christopher Jeffrey. (MIT License)
   * https://github.com/markedjs/marked
   */

  /**
   * DO NOT EDIT THIS FILE
   * The code in this file is generated from files in ./src/
   */

  function M(){return {async:false,breaks:false,extensions:null,gfm:true,hooks:null,pedantic:false,renderer:null,silent:false,tokenizer:null,walkTokens:null}}var T=M();function N(l){T=l;}var _={exec:()=>null};function E(l){let e=[];return t=>{let n=Math.max(0,Math.min(3,t-1)),s=e[n];return s||(s=l(n),e[n]=s),s}}function d(l,e=""){let t=typeof l=="string"?l:l.source,n={replace:(s,r)=>{let i=typeof r=="string"?r:r.source;return i=i.replace(m.caret,"$1"),t=t.replace(s,i),n},getRegex:()=>new RegExp(t,e)};return n}var Te=((l="")=>{try{return !!new RegExp("(?<=1)(?<!1)"+l)}catch{return  false}})(),m={codeRemoveIndent:/^(?: {1,4}| {0,3}\t)/gm,outputLinkReplace:/\\([\[\]])/g,indentCodeCompensation:/^(\s+)(?:```)/,beginningSpace:/^\s+/,endingHash:/#$/,startingSpaceChar:/^ /,endingSpaceChar:/ $/,nonSpaceChar:/[^ ]/,newLineCharGlobal:/\n/g,tabCharGlobal:/\t/g,multipleSpaceGlobal:/\s+/g,blankLine:/^[ \t]*$/,doubleBlankLine:/\n[ \t]*\n[ \t]*$/,blockquoteStart:/^ {0,3}>/,blockquoteSetextReplace:/\n {0,3}((?:=+|-+) *)(?=\n|$)/g,blockquoteSetextReplace2:/^ {0,3}>[ \t]?/gm,listReplaceNesting:/^ {1,4}(?=( {4})*[^ ])/g,listIsTask:/^\[[ xX]\] +\S/,listReplaceTask:/^\[[ xX]\] +/,listTaskCheckbox:/\[[ xX]\]/,anyLine:/\n.*\n/,hrefBrackets:/^<(.*)>$/,tableDelimiter:/[:|]/,tableAlignChars:/^\||\| *$/g,tableRowBlankLine:/\n[ \t]*$/,tableAlignRight:/^ *-+: *$/,tableAlignCenter:/^ *:-+: *$/,tableAlignLeft:/^ *:-+ *$/,startATag:/^<a /i,endATag:/^<\/a>/i,startPreScriptTag:/^<(pre|code|kbd|script)(\s|>)/i,endPreScriptTag:/^<\/(pre|code|kbd|script)(\s|>)/i,startAngleBracket:/^</,endAngleBracket:/>$/,pedanticHrefTitle:/^([^'"]*[^\s])\s+(['"])(.*)\2/,unicodeAlphaNumeric:/[\p{L}\p{N}]/u,escapeTest:/[&<>"']/,escapeReplace:/[&<>"']/g,escapeTestNoEncode:/[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/,escapeReplaceNoEncode:/[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/g,caret:/(^|[^\[])\^/g,percentDecode:/%25/g,findPipe:/\|/g,splitPipe:/ \|/,slashPipe:/\\\|/g,carriageReturn:/\r\n|\r/g,spaceLine:/^ +$/gm,notSpaceStart:/^\S*/,endingNewline:/\n$/,listItemRegex:l=>new RegExp(`^( {0,3}${l})((?:[	 ][^\\n]*)?(?:\\n|$))`),nextBulletRegex:E(l=>new RegExp(`^ {0,${l}}(?:[*+-]|\\d{1,9}[.)])((?:[ 	][^\\n]*)?(?:\\n|$))`)),hrRegex:E(l=>new RegExp(`^ {0,${l}}((?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$)`)),fencesBeginRegex:E(l=>new RegExp(`^ {0,${l}}(?:\`\`\`|~~~)`)),headingBeginRegex:E(l=>new RegExp(`^ {0,${l}}#`)),htmlBeginRegex:E(l=>new RegExp(`^ {0,${l}}<(?:[a-z].*>|!--)`,"i")),blockquoteBeginRegex:E(l=>new RegExp(`^ {0,${l}}>`))},Oe=/^(?:[ \t]*(?:\n|$))+/,we=/^((?: {4}| {0,3}\t)[^\n]+(?:\n(?:[ \t]*(?:\n|$))*)?)+/,ye=/^ {0,3}(`{3,}(?=[^`\n]*(?:\n|$))|~{3,})([^\n]*)(?:\n|$)(?:|([\s\S]*?)(?:\n|$))(?: {0,3}\1[~`]* *(?=\n|$)|$)/,B=/^ {0,3}((?:-[\t ]*){3,}|(?:_[ \t]*){3,}|(?:\*[ \t]*){3,})(?:\n+|$)/,Pe=/^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/,j=/ {0,3}(?:[*+-]|\d{1,9}[.)])/,oe=/^(?!bull |blockCode|fences|blockquote|heading|html|table)((?:.|\n(?!\s*?\n|bull |blockCode|fences|blockquote|heading|html|table))+?)\n {0,3}(=+|-+) *(?:\n+|$)/,ae=d(oe).replace(/bull/g,j).replace(/blockCode/g,/(?: {4}| {0,3}\t)/).replace(/fences/g,/ {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g,/ {0,3}>/).replace(/heading/g,/ {0,3}#{1,6}/).replace(/html/g,/ {0,3}<[^\n>]+>\n/).replace(/\|table/g,"").getRegex(),Se=d(oe).replace(/bull/g,j).replace(/blockCode/g,/(?: {4}| {0,3}\t)/).replace(/fences/g,/ {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g,/ {0,3}>/).replace(/heading/g,/ {0,3}#{1,6}/).replace(/html/g,/ {0,3}<[^\n>]+>\n/).replace(/table/g,/ {0,3}\|?(?:[:\- ]*\|)+[\:\- ]*\n/).getRegex(),F=/^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/,$e=/^[^\n]+/,U=/(?!\s*\])(?:\\[\s\S]|[^\[\]\\])+/,Le=d(/^ {0,3}\[(label)\]: *(?:\n[ \t]*)?([^<\s][^\s]*|<.*?>)(?:(?: +(?:\n[ \t]*)?| *\n[ \t]*)(title))? *(?:\n+|$)/).replace("label",U).replace("title",/(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/).getRegex(),_e=d(/^(bull)([ \t][^\n]*?)?(?:\n|$)/).replace(/bull/g,j).getRegex(),H="address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul",K=/<!--(?:-?>|[\s\S]*?(?:-->|$))/,ze=d("^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n[ 	]*)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ 	]*)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ 	]*)+\\n|$))","i").replace("comment",K).replace("tag",H).replace("attribute",/ +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/).getRegex(),le=d(F).replace("hr",B).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("|lheading","").replace("|table","").replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)])[ \\t]+[^ \\t\\n]").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",H).getRegex(),Me=d(/^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/).replace("paragraph",le).getRegex(),W={blockquote:Me,code:we,def:Le,fences:ye,heading:Pe,hr:B,html:ze,lheading:ae,list:_e,newline:Oe,paragraph:le,table:_,text:$e},se=d("^ *([^\\n ].*)\\n {0,3}((?:\\| *)?:?-+:? *(?:\\| *:?-+:? *)*(?:\\| *)?)(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)").replace("hr",B).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("blockquote"," {0,3}>").replace("code","(?: {4}| {0,3}	)[^\\n]").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)])[ \\t]").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",H).getRegex(),Ee={...W,lheading:Se,table:se,paragraph:d(F).replace("hr",B).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("|lheading","").replace("table",se).replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)])[ \\t]+[^ \\t\\n]").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",H).getRegex()},Ie={...W,html:d(`^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:"[^"]*"|'[^']*'|\\s[^'"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))`).replace("comment",K).replace(/tag/g,"(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b").getRegex(),def:/^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/,heading:/^(#{1,6})(.*)(?:\n+|$)/,fences:_,lheading:/^(.+?)\n {0,3}(=+|-+) *(?:\n+|$)/,paragraph:d(F).replace("hr",B).replace("heading",` *#{1,6} *[^
]`).replace("lheading",ae).replace("|table","").replace("blockquote"," {0,3}>").replace("|fences","").replace("|list","").replace("|html","").replace("|tag","").getRegex()},Ae=/^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/,Ce=/^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/,ue=/^( {2,}|\\)\n(?!\s*$)/,Be=/^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/,I=/[\p{P}\p{S}]/u,Z=/[\s\p{P}\p{S}]/u,X=/[^\s\p{P}\p{S}]/u,De=d(/^((?![*_])punctSpace)/,"u").replace(/punctSpace/g,Z).getRegex(),pe=/(?!~)[\p{P}\p{S}]/u,qe=/(?!~)[\s\p{P}\p{S}]/u,ve=/(?:[^\s\p{P}\p{S}]|~)/u,He=d(/link|precode-code|html/,"g").replace("link",/\[(?:[^\[\]`]|(?<a>`+)[^`]+\k<a>(?!`))*?\]\((?:\\[\s\S]|[^\\\(\)]|\((?:\\[\s\S]|[^\\\(\)])*\))*\)/).replace("precode-",Te?"(?<!`)()":"(^^|[^`])").replace("code",/(?<b>`+)[^`]+\k<b>(?!`)/).replace("html",/<(?! )[^<>]*?>/).getRegex(),ce=/^(?:\*+(?:((?!\*)punct)|([^\s*]))?)|^_+(?:((?!_)punct)|([^\s_]))?/,Ze=d(ce,"u").replace(/punct/g,I).getRegex(),Ge=d(ce,"u").replace(/punct/g,pe).getRegex(),he="^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)punct(\\*+)(?=[\\s]|$)|notPunctSpace(\\*+)(?!\\*)(?=punctSpace|$)|(?!\\*)punctSpace(\\*+)(?=notPunctSpace)|[\\s](\\*+)(?!\\*)(?=punct)|(?!\\*)punct(\\*+)(?!\\*)(?=punct)|notPunctSpace(\\*+)(?=notPunctSpace)",Ne=d(he,"gu").replace(/notPunctSpace/g,X).replace(/punctSpace/g,Z).replace(/punct/g,I).getRegex(),Qe=d(he,"gu").replace(/notPunctSpace/g,ve).replace(/punctSpace/g,qe).replace(/punct/g,pe).getRegex(),je=d("^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)punct(_+)(?=[\\s]|$)|notPunctSpace(_+)(?!_)(?=punctSpace|$)|(?!_)punctSpace(_+)(?=notPunctSpace)|[\\s](_+)(?!_)(?=punct)|(?!_)punct(_+)(?!_)(?=punct)","gu").replace(/notPunctSpace/g,X).replace(/punctSpace/g,Z).replace(/punct/g,I).getRegex(),Fe=d(/^~~?(?:((?!~)punct)|[^\s~])/,"u").replace(/punct/g,I).getRegex(),Ue="^[^~]+(?=[^~])|(?!~)punct(~~?)(?=[\\s]|$)|notPunctSpace(~~?)(?!~)(?=punctSpace|$)|(?!~)punctSpace(~~?)(?=notPunctSpace)|[\\s](~~?)(?!~)(?=punct)|(?!~)punct(~~?)(?!~)(?=punct)|notPunctSpace(~~?)(?=notPunctSpace)",Ke=d(Ue,"gu").replace(/notPunctSpace/g,X).replace(/punctSpace/g,Z).replace(/punct/g,I).getRegex(),We=d(/\\(punct)/,"gu").replace(/punct/g,I).getRegex(),Xe=d(/^<(scheme:[^\s\x00-\x1f<>]*|email)>/).replace("scheme",/[a-zA-Z][a-zA-Z0-9+.-]{1,31}/).replace("email",/[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/).getRegex(),Je=d(K).replace("(?:-->|$)","-->").getRegex(),Ve=d("^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>").replace("comment",Je).replace("attribute",/\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/).getRegex(),v=/(?:\[(?:\\[\s\S]|[^\[\]\\])*\]|\\[\s\S]|`+(?!`)[^`]*?`+(?!`)|``+(?=\])|[^\[\]\\`])*?/,Ye=d(/^!?\[(label)\]\(\s*(href)(?:(?:[ \t]+(?:\n[ \t]*)?|\n[ \t]*)(title))?\s*\)/).replace("label",v).replace("href",/<(?:\\.|[^\n<>\\])+>|[^ \t\n\x00-\x1f]*/).replace("title",/"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/).getRegex(),ke=d(/^!?\[(label)\]\[(ref)\]/).replace("label",v).replace("ref",U).getRegex(),de=d(/^!?\[(ref)\](?:\[\])?/).replace("ref",U).getRegex(),et=d("reflink|nolink(?!\\()","g").replace("reflink",ke).replace("nolink",de).getRegex(),ie=/[hH][tT][tT][pP][sS]?|[fF][tT][pP]/,J={_backpedal:_,anyPunctuation:We,autolink:Xe,blockSkip:He,br:ue,code:Ce,del:_,delLDelim:_,delRDelim:_,emStrongLDelim:Ze,emStrongRDelimAst:Ne,emStrongRDelimUnd:je,escape:Ae,link:Ye,nolink:de,punctuation:De,reflink:ke,reflinkSearch:et,tag:Ve,text:Be,url:_},tt={...J,link:d(/^!?\[(label)\]\((.*?)\)/).replace("label",v).getRegex(),reflink:d(/^!?\[(label)\]\s*\[([^\]]*)\]/).replace("label",v).getRegex()},Q={...J,emStrongRDelimAst:Qe,emStrongLDelim:Ge,delLDelim:Fe,delRDelim:Ke,url:d(/^((?:protocol):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/).replace("protocol",ie).replace("email",/[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/).getRegex(),_backpedal:/(?:[^?!.,:;*_'"~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_'"~)]+(?!$))+/,del:/^(~~?)(?=[^\s~])((?:\\[\s\S]|[^\\])*?(?:\\[\s\S]|[^\s~\\]))\1(?=[^~]|$)/,text:d(/^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|protocol:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/).replace("protocol",ie).getRegex()},nt={...Q,br:d(ue).replace("{2,}","*").getRegex(),text:d(Q.text).replace("\\b_","\\b_| {2,}\\n").replace(/\{2,\}/g,"*").getRegex()},D={normal:W,gfm:Ee,pedantic:Ie},A={normal:J,gfm:Q,breaks:nt,pedantic:tt};var rt={"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"},ge=l=>rt[l];function O(l,e){if(e){if(m.escapeTest.test(l))return l.replace(m.escapeReplace,ge)}else if(m.escapeTestNoEncode.test(l))return l.replace(m.escapeReplaceNoEncode,ge);return l}function V(l){try{l=encodeURI(l).replace(m.percentDecode,"%");}catch{return null}return l}function Y(l,e){let t=l.replace(m.findPipe,(r,i,o)=>{let u=false,a=i;for(;--a>=0&&o[a]==="\\";)u=!u;return u?"|":" |"}),n=t.split(m.splitPipe),s=0;if(n[0].trim()||n.shift(),n.length>0&&!n.at(-1)?.trim()&&n.pop(),e)if(n.length>e)n.splice(e);else for(;n.length<e;)n.push("");for(;s<n.length;s++)n[s]=n[s].trim().replace(m.slashPipe,"|");return n}function $(l,e,t){let n=l.length;if(n===0)return "";let s=0;for(;s<n;){let r=l.charAt(n-s-1);if(r===e&&true)s++;else break}return l.slice(0,n-s)}function ee(l){let e=l.split(`
`),t=e.length-1;for(;t>=0&&m.blankLine.test(e[t]);)t--;return e.length-t<=2?l:e.slice(0,t+1).join(`
`)}function fe(l,e){if(l.indexOf(e[1])===-1)return  -1;let t=0;for(let n=0;n<l.length;n++)if(l[n]==="\\")n++;else if(l[n]===e[0])t++;else if(l[n]===e[1]&&(t--,t<0))return n;return t>0?-2:-1}function me(l,e=0){let t=e,n="";for(let s of l)if(s==="	"){let r=4-t%4;n+=" ".repeat(r),t+=r;}else n+=s,t++;return n}function xe(l,e,t,n,s){let r=e.href,i=e.title||null,o=l[1].replace(s.other.outputLinkReplace,"$1");n.state.inLink=true;let u={type:l[0].charAt(0)==="!"?"image":"link",raw:t,href:r,title:i,text:o,tokens:n.inlineTokens(o)};return n.state.inLink=false,u}function st(l,e,t){let n=l.match(t.other.indentCodeCompensation);if(n===null)return e;let s=n[1];return e.split(`
`).map(r=>{let i=r.match(t.other.beginningSpace);if(i===null)return r;let[o]=i;return o.length>=s.length?r.slice(s.length):r}).join(`
`)}var w=class{options;rules;lexer;constructor(e){this.options=e||T;}space(e){let t=this.rules.block.newline.exec(e);if(t&&t[0].length>0)return {type:"space",raw:t[0]}}code(e){let t=this.rules.block.code.exec(e);if(t){let n=this.options.pedantic?t[0]:ee(t[0]),s=n.replace(this.rules.other.codeRemoveIndent,"");return {type:"code",raw:n,codeBlockStyle:"indented",text:s}}}fences(e){let t=this.rules.block.fences.exec(e);if(t){let n=t[0],s=st(n,t[3]||"",this.rules);return {type:"code",raw:n,lang:t[2]?t[2].trim().replace(this.rules.inline.anyPunctuation,"$1"):t[2],text:s}}}heading(e){let t=this.rules.block.heading.exec(e);if(t){let n=t[2].trim();if(this.rules.other.endingHash.test(n)){let s=$(n,"#");(this.options.pedantic||!s||this.rules.other.endingSpaceChar.test(s))&&(n=s.trim());}return {type:"heading",raw:$(t[0],`
`),depth:t[1].length,text:n,tokens:this.lexer.inline(n)}}}hr(e){let t=this.rules.block.hr.exec(e);if(t)return {type:"hr",raw:$(t[0],`
`)}}blockquote(e){let t=this.rules.block.blockquote.exec(e);if(t){let n=$(t[0],`
`).split(`
`),s="",r="",i=[];for(;n.length>0;){let o=false,u=[],a;for(a=0;a<n.length;a++)if(this.rules.other.blockquoteStart.test(n[a]))u.push(n[a]),o=true;else if(!o)u.push(n[a]);else break;n=n.slice(a);let c=u.join(`
`),p=c.replace(this.rules.other.blockquoteSetextReplace,`
    $1`).replace(this.rules.other.blockquoteSetextReplace2,"");s=s?`${s}
${c}`:c,r=r?`${r}
${p}`:p;let k=this.lexer.state.top;if(this.lexer.state.top=true,this.lexer.blockTokens(p,i,true),this.lexer.state.top=k,n.length===0)break;let h=i.at(-1);if(h?.type==="code")break;if(h?.type==="blockquote"){let R=h,f=R.raw+`
`+n.join(`
`),S=this.blockquote(f);i[i.length-1]=S,s=s.substring(0,s.length-R.raw.length)+S.raw,r=r.substring(0,r.length-R.text.length)+S.text;break}else if(h?.type==="list"){let R=h,f=R.raw+`
`+n.join(`
`),S=this.list(f);i[i.length-1]=S,s=s.substring(0,s.length-h.raw.length)+S.raw,r=r.substring(0,r.length-R.raw.length)+S.raw,n=f.substring(i.at(-1).raw.length).split(`
`);continue}}return {type:"blockquote",raw:s,tokens:i,text:r}}}list(e){let t=this.rules.block.list.exec(e);if(t){let n=t[1].trim(),s=n.length>1,r={type:"list",raw:"",ordered:s,start:s?+n.slice(0,-1):"",loose:false,items:[]};n=s?`\\d{1,9}\\${n.slice(-1)}`:`\\${n}`,this.options.pedantic&&(n=s?n:"[*+-]");let i=this.rules.other.listItemRegex(n),o=false;for(;e;){let a=false,c="",p="";if(!(t=i.exec(e))||this.rules.block.hr.test(e))break;c=t[0],e=e.substring(c.length);let k=me(t[2].split(`
`,1)[0],t[1].length),h=e.split(`
`,1)[0],R=!k.trim(),f=0;if(this.options.pedantic?(f=2,p=k.trimStart()):R?f=t[1].length+1:(f=k.search(this.rules.other.nonSpaceChar),f=f>4?1:f,p=k.slice(f),f+=t[1].length),R&&this.rules.other.blankLine.test(h)&&(c+=h+`
`,e=e.substring(h.length+1),a=true),!a){let S=this.rules.other.nextBulletRegex(f),te=this.rules.other.hrRegex(f),ne=this.rules.other.fencesBeginRegex(f),re=this.rules.other.headingBeginRegex(f),be=this.rules.other.htmlBeginRegex(f),Re=this.rules.other.blockquoteBeginRegex(f);for(;e;){let G=e.split(`
`,1)[0],C;if(h=G,this.options.pedantic?(h=h.replace(this.rules.other.listReplaceNesting,"  "),C=h):C=h.replace(this.rules.other.tabCharGlobal,"    "),ne.test(h)||re.test(h)||be.test(h)||Re.test(h)||S.test(h)||te.test(h))break;if(C.search(this.rules.other.nonSpaceChar)>=f||!h.trim())p+=`
`+C.slice(f);else {if(R||k.replace(this.rules.other.tabCharGlobal,"    ").search(this.rules.other.nonSpaceChar)>=4||ne.test(k)||re.test(k)||te.test(k))break;p+=`
`+h;}R=!h.trim(),c+=G+`
`,e=e.substring(G.length+1),k=C.slice(f);}}r.loose||(o?r.loose=true:this.rules.other.doubleBlankLine.test(c)&&(o=true)),r.items.push({type:"list_item",raw:c,task:!!this.options.gfm&&this.rules.other.listIsTask.test(p),loose:false,text:p,tokens:[]}),r.raw+=c;}let u=r.items.at(-1);if(u)u.raw=u.raw.trimEnd(),u.text=u.text.trimEnd();else return;r.raw=r.raw.trimEnd();for(let a of r.items){this.lexer.state.top=false,a.tokens=this.lexer.blockTokens(a.text,[]);let c=a.tokens[0];if(a.task&&(c?.type==="text"||c?.type==="paragraph")){a.text=a.text.replace(this.rules.other.listReplaceTask,""),c.raw=c.raw.replace(this.rules.other.listReplaceTask,""),c.text=c.text.replace(this.rules.other.listReplaceTask,"");for(let k=this.lexer.inlineQueue.length-1;k>=0;k--)if(this.rules.other.listIsTask.test(this.lexer.inlineQueue[k].src)){this.lexer.inlineQueue[k].src=this.lexer.inlineQueue[k].src.replace(this.rules.other.listReplaceTask,"");break}let p=this.rules.other.listTaskCheckbox.exec(a.raw);if(p){let k={type:"checkbox",raw:p[0]+" ",checked:p[0]!=="[ ]"};a.checked=k.checked,r.loose?a.tokens[0]&&["paragraph","text"].includes(a.tokens[0].type)&&"tokens"in a.tokens[0]&&a.tokens[0].tokens?(a.tokens[0].raw=k.raw+a.tokens[0].raw,a.tokens[0].text=k.raw+a.tokens[0].text,a.tokens[0].tokens.unshift(k)):a.tokens.unshift({type:"paragraph",raw:k.raw,text:k.raw,tokens:[k]}):a.tokens.unshift(k);}}else a.task&&(a.task=false);if(!r.loose){let p=a.tokens.filter(h=>h.type==="space"),k=p.length>0&&p.some(h=>this.rules.other.anyLine.test(h.raw));r.loose=k;}}if(r.loose)for(let a of r.items){a.loose=true;for(let c of a.tokens)c.type==="text"&&(c.type="paragraph");}return r}}html(e){let t=this.rules.block.html.exec(e);if(t){let n=ee(t[0]);return {type:"html",block:true,raw:n,pre:t[1]==="pre"||t[1]==="script"||t[1]==="style",text:n}}}def(e){let t=this.rules.block.def.exec(e);if(t){let n=t[1].toLowerCase().replace(this.rules.other.multipleSpaceGlobal," "),s=t[2]?t[2].replace(this.rules.other.hrefBrackets,"$1").replace(this.rules.inline.anyPunctuation,"$1"):"",r=t[3]?t[3].substring(1,t[3].length-1).replace(this.rules.inline.anyPunctuation,"$1"):t[3];return {type:"def",tag:n,raw:$(t[0],`
`),href:s,title:r}}}table(e){let t=this.rules.block.table.exec(e);if(!t||!this.rules.other.tableDelimiter.test(t[2]))return;let n=Y(t[1]),s=t[2].replace(this.rules.other.tableAlignChars,"").split("|"),r=t[3]?.trim()?t[3].replace(this.rules.other.tableRowBlankLine,"").split(`
`):[],i={type:"table",raw:$(t[0],`
`),header:[],align:[],rows:[]};if(n.length===s.length){for(let o of s)this.rules.other.tableAlignRight.test(o)?i.align.push("right"):this.rules.other.tableAlignCenter.test(o)?i.align.push("center"):this.rules.other.tableAlignLeft.test(o)?i.align.push("left"):i.align.push(null);for(let o=0;o<n.length;o++)i.header.push({text:n[o],tokens:this.lexer.inline(n[o]),header:true,align:i.align[o]});for(let o of r)i.rows.push(Y(o,i.header.length).map((u,a)=>({text:u,tokens:this.lexer.inline(u),header:false,align:i.align[a]})));return i}}lheading(e){let t=this.rules.block.lheading.exec(e);if(t){let n=t[1].trim();return {type:"heading",raw:$(t[0],`
`),depth:t[2].charAt(0)==="="?1:2,text:n,tokens:this.lexer.inline(n)}}}paragraph(e){let t=this.rules.block.paragraph.exec(e);if(t){let n=t[1].charAt(t[1].length-1)===`
`?t[1].slice(0,-1):t[1];return {type:"paragraph",raw:t[0],text:n,tokens:this.lexer.inline(n)}}}text(e){let t=this.rules.block.text.exec(e);if(t)return {type:"text",raw:t[0],text:t[0],tokens:this.lexer.inline(t[0])}}escape(e){let t=this.rules.inline.escape.exec(e);if(t)return {type:"escape",raw:t[0],text:t[1]}}tag(e){let t=this.rules.inline.tag.exec(e);if(t)return !this.lexer.state.inLink&&this.rules.other.startATag.test(t[0])?this.lexer.state.inLink=true:this.lexer.state.inLink&&this.rules.other.endATag.test(t[0])&&(this.lexer.state.inLink=false),!this.lexer.state.inRawBlock&&this.rules.other.startPreScriptTag.test(t[0])?this.lexer.state.inRawBlock=true:this.lexer.state.inRawBlock&&this.rules.other.endPreScriptTag.test(t[0])&&(this.lexer.state.inRawBlock=false),{type:"html",raw:t[0],inLink:this.lexer.state.inLink,inRawBlock:this.lexer.state.inRawBlock,block:false,text:t[0]}}link(e){let t=this.rules.inline.link.exec(e);if(t){let n=t[2].trim();if(!this.options.pedantic&&this.rules.other.startAngleBracket.test(n)){if(!this.rules.other.endAngleBracket.test(n))return;let i=$(n.slice(0,-1),"\\");if((n.length-i.length)%2===0)return}else {let i=fe(t[2],"()");if(i===-2)return;if(i>-1){let u=(t[0].indexOf("!")===0?5:4)+t[1].length+i;t[2]=t[2].substring(0,i),t[0]=t[0].substring(0,u).trim(),t[3]="";}}let s=t[2],r="";if(this.options.pedantic){let i=this.rules.other.pedanticHrefTitle.exec(s);i&&(s=i[1],r=i[3]);}else r=t[3]?t[3].slice(1,-1):"";return s=s.trim(),this.rules.other.startAngleBracket.test(s)&&(this.options.pedantic&&!this.rules.other.endAngleBracket.test(n)?s=s.slice(1):s=s.slice(1,-1)),xe(t,{href:s&&s.replace(this.rules.inline.anyPunctuation,"$1"),title:r&&r.replace(this.rules.inline.anyPunctuation,"$1")},t[0],this.lexer,this.rules)}}reflink(e,t){let n;if((n=this.rules.inline.reflink.exec(e))||(n=this.rules.inline.nolink.exec(e))){let s=(n[2]||n[1]).replace(this.rules.other.multipleSpaceGlobal," "),r=t[s.toLowerCase()];if(!r){let i=n[0].charAt(0);return {type:"text",raw:i,text:i}}return xe(n,r,n[0],this.lexer,this.rules)}}emStrong(e,t,n=""){let s=this.rules.inline.emStrongLDelim.exec(e);if(!s||!s[1]&&!s[2]&&!s[3]&&!s[4]||s[4]&&n.match(this.rules.other.unicodeAlphaNumeric))return;if(!(s[1]||s[3]||"")||!n||this.rules.inline.punctuation.exec(n)){let i=[...s[0]].length-1,o,u,a=i,c=0,p=s[0][0]==="*"?this.rules.inline.emStrongRDelimAst:this.rules.inline.emStrongRDelimUnd;for(p.lastIndex=0,t=t.slice(-1*e.length+i);(s=p.exec(t))!==null;){if(o=s[1]||s[2]||s[3]||s[4]||s[5]||s[6],!o)continue;if(u=[...o].length,s[3]||s[4]){a+=u;continue}else if((s[5]||s[6])&&i%3&&!((i+u)%3)){c+=u;continue}if(a-=u,a>0)continue;u=Math.min(u,u+a+c);let k=[...s[0]][0].length,h=e.slice(0,i+s.index+k+u);if(Math.min(i,u)%2){let f=h.slice(1,-1);return {type:"em",raw:h,text:f,tokens:this.lexer.inlineTokens(f)}}let R=h.slice(2,-2);return {type:"strong",raw:h,text:R,tokens:this.lexer.inlineTokens(R)}}}}codespan(e){let t=this.rules.inline.code.exec(e);if(t){let n=t[2].replace(this.rules.other.newLineCharGlobal," "),s=this.rules.other.nonSpaceChar.test(n),r=this.rules.other.startingSpaceChar.test(n)&&this.rules.other.endingSpaceChar.test(n);return s&&r&&(n=n.substring(1,n.length-1)),{type:"codespan",raw:t[0],text:n}}}br(e){let t=this.rules.inline.br.exec(e);if(t)return {type:"br",raw:t[0]}}del(e,t,n=""){let s=this.rules.inline.delLDelim.exec(e);if(!s)return;if(!(s[1]||"")||!n||this.rules.inline.punctuation.exec(n)){let i=[...s[0]].length-1,o,u,a=i,c=this.rules.inline.delRDelim;for(c.lastIndex=0,t=t.slice(-1*e.length+i);(s=c.exec(t))!==null;){if(o=s[1]||s[2]||s[3]||s[4]||s[5]||s[6],!o||(u=[...o].length,u!==i))continue;if(s[3]||s[4]){a+=u;continue}if(a-=u,a>0)continue;u=Math.min(u,u+a);let p=[...s[0]][0].length,k=e.slice(0,i+s.index+p+u),h=k.slice(i,-i);return {type:"del",raw:k,text:h,tokens:this.lexer.inlineTokens(h)}}}}autolink(e){let t=this.rules.inline.autolink.exec(e);if(t){let n,s;return t[2]==="@"?(n=t[1],s="mailto:"+n):(n=t[1],s=n),{type:"link",raw:t[0],text:n,href:s,tokens:[{type:"text",raw:n,text:n}]}}}url(e){let t;if(t=this.rules.inline.url.exec(e)){let n,s;if(t[2]==="@")n=t[0],s="mailto:"+n;else {let r;do r=t[0],t[0]=this.rules.inline._backpedal.exec(t[0])?.[0]??"";while(r!==t[0]);n=t[0],t[1]==="www."?s="http://"+t[0]:s=t[0];}return {type:"link",raw:t[0],text:n,href:s,tokens:[{type:"text",raw:n,text:n}]}}}inlineText(e){let t=this.rules.inline.text.exec(e);if(t){let n=this.lexer.state.inRawBlock;return {type:"text",raw:t[0],text:t[0],escaped:n}}}};var x=class l{tokens;options;state;inlineQueue;tokenizer;constructor(e){this.tokens=[],this.tokens.links=Object.create(null),this.options=e||T,this.options.tokenizer=this.options.tokenizer||new w,this.tokenizer=this.options.tokenizer,this.tokenizer.options=this.options,this.tokenizer.lexer=this,this.inlineQueue=[],this.state={inLink:false,inRawBlock:false,top:true};let t={other:m,block:D.normal,inline:A.normal};this.options.pedantic?(t.block=D.pedantic,t.inline=A.pedantic):this.options.gfm&&(t.block=D.gfm,this.options.breaks?t.inline=A.breaks:t.inline=A.gfm),this.tokenizer.rules=t;}static get rules(){return {block:D,inline:A}}static lex(e,t){return new l(t).lex(e)}static lexInline(e,t){return new l(t).inlineTokens(e)}lex(e){e=e.replace(m.carriageReturn,`
`),this.blockTokens(e,this.tokens);for(let t=0;t<this.inlineQueue.length;t++){let n=this.inlineQueue[t];this.inlineTokens(n.src,n.tokens);}return this.inlineQueue=[],this.tokens}blockTokens(e,t=[],n=false){this.tokenizer.lexer=this,this.options.pedantic&&(e=e.replace(m.tabCharGlobal,"    ").replace(m.spaceLine,""));let s=1/0;for(;e;){if(e.length<s)s=e.length;else {this.infiniteLoopError(e.charCodeAt(0));break}let r;if(this.options.extensions?.block?.some(o=>(r=o.call({lexer:this},e,t))?(e=e.substring(r.raw.length),t.push(r),true):false))continue;if(r=this.tokenizer.space(e)){e=e.substring(r.raw.length);let o=t.at(-1);r.raw.length===1&&o!==void 0?o.raw+=`
`:t.push(r);continue}if(r=this.tokenizer.code(e)){e=e.substring(r.raw.length);let o=t.at(-1);o?.type==="paragraph"||o?.type==="text"?(o.raw+=(o.raw.endsWith(`
`)?"":`
`)+r.raw,o.text+=`
`+r.text,this.inlineQueue.at(-1).src=o.text):t.push(r);continue}if(r=this.tokenizer.fences(e)){e=e.substring(r.raw.length),t.push(r);continue}if(r=this.tokenizer.heading(e)){e=e.substring(r.raw.length),t.push(r);continue}if(r=this.tokenizer.hr(e)){e=e.substring(r.raw.length),t.push(r);continue}if(r=this.tokenizer.blockquote(e)){e=e.substring(r.raw.length),t.push(r);continue}if(r=this.tokenizer.list(e)){e=e.substring(r.raw.length),t.push(r);continue}if(r=this.tokenizer.html(e)){e=e.substring(r.raw.length),t.push(r);continue}if(r=this.tokenizer.def(e)){e=e.substring(r.raw.length);let o=t.at(-1);o?.type==="paragraph"||o?.type==="text"?(o.raw+=(o.raw.endsWith(`
`)?"":`
`)+r.raw,o.text+=`
`+r.raw,this.inlineQueue.at(-1).src=o.text):this.tokens.links[r.tag]||(this.tokens.links[r.tag]={href:r.href,title:r.title},t.push(r));continue}if(r=this.tokenizer.table(e)){e=e.substring(r.raw.length),t.push(r);continue}if(r=this.tokenizer.lheading(e)){e=e.substring(r.raw.length),t.push(r);continue}let i=e;if(this.options.extensions?.startBlock){let o=1/0,u=e.slice(1),a;this.options.extensions.startBlock.forEach(c=>{a=c.call({lexer:this},u),typeof a=="number"&&a>=0&&(o=Math.min(o,a));}),o<1/0&&o>=0&&(i=e.substring(0,o+1));}if(this.state.top&&(r=this.tokenizer.paragraph(i))){let o=t.at(-1);n&&o?.type==="paragraph"?(o.raw+=(o.raw.endsWith(`
`)?"":`
`)+r.raw,o.text+=`
`+r.text,this.inlineQueue.pop(),this.inlineQueue.at(-1).src=o.text):t.push(r),n=i.length!==e.length,e=e.substring(r.raw.length);continue}if(r=this.tokenizer.text(e)){e=e.substring(r.raw.length);let o=t.at(-1);o?.type==="text"?(o.raw+=(o.raw.endsWith(`
`)?"":`
`)+r.raw,o.text+=`
`+r.text,this.inlineQueue.pop(),this.inlineQueue.at(-1).src=o.text):t.push(r);continue}if(e){this.infiniteLoopError(e.charCodeAt(0));break}}return this.state.top=true,t}inline(e,t=[]){return this.inlineQueue.push({src:e,tokens:t}),t}inlineTokens(e,t=[]){this.tokenizer.lexer=this;let n=e,s=null;if(this.tokens.links){let a=Object.keys(this.tokens.links);if(a.length>0)for(;(s=this.tokenizer.rules.inline.reflinkSearch.exec(n))!==null;)a.includes(s[0].slice(s[0].lastIndexOf("[")+1,-1))&&(n=n.slice(0,s.index)+"["+"a".repeat(s[0].length-2)+"]"+n.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex));}for(;(s=this.tokenizer.rules.inline.anyPunctuation.exec(n))!==null;)n=n.slice(0,s.index)+"++"+n.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);let r;for(;(s=this.tokenizer.rules.inline.blockSkip.exec(n))!==null;)r=s[2]?s[2].length:0,n=n.slice(0,s.index+r)+"["+"a".repeat(s[0].length-r-2)+"]"+n.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);n=this.options.hooks?.emStrongMask?.call({lexer:this},n)??n;let i=false,o="",u=1/0;for(;e;){if(e.length<u)u=e.length;else {this.infiniteLoopError(e.charCodeAt(0));break}i||(o=""),i=false;let a;if(this.options.extensions?.inline?.some(p=>(a=p.call({lexer:this},e,t))?(e=e.substring(a.raw.length),t.push(a),true):false))continue;if(a=this.tokenizer.escape(e)){e=e.substring(a.raw.length),t.push(a);continue}if(a=this.tokenizer.tag(e)){e=e.substring(a.raw.length),t.push(a);continue}if(a=this.tokenizer.link(e)){e=e.substring(a.raw.length),t.push(a);continue}if(a=this.tokenizer.reflink(e,this.tokens.links)){e=e.substring(a.raw.length);let p=t.at(-1);a.type==="text"&&p?.type==="text"?(p.raw+=a.raw,p.text+=a.text):t.push(a);continue}if(a=this.tokenizer.emStrong(e,n,o)){e=e.substring(a.raw.length),t.push(a);continue}if(a=this.tokenizer.codespan(e)){e=e.substring(a.raw.length),t.push(a);continue}if(a=this.tokenizer.br(e)){e=e.substring(a.raw.length),t.push(a);continue}if(a=this.tokenizer.del(e,n,o)){e=e.substring(a.raw.length),t.push(a);continue}if(a=this.tokenizer.autolink(e)){e=e.substring(a.raw.length),t.push(a);continue}if(!this.state.inLink&&(a=this.tokenizer.url(e))){e=e.substring(a.raw.length),t.push(a);continue}let c=e;if(this.options.extensions?.startInline){let p=1/0,k=e.slice(1),h;this.options.extensions.startInline.forEach(R=>{h=R.call({lexer:this},k),typeof h=="number"&&h>=0&&(p=Math.min(p,h));}),p<1/0&&p>=0&&(c=e.substring(0,p+1));}if(a=this.tokenizer.inlineText(c)){e=e.substring(a.raw.length),a.raw.slice(-1)!=="_"&&(o=a.raw.slice(-1)),i=true;let p=t.at(-1);p?.type==="text"?(p.raw+=a.raw,p.text+=a.text):t.push(a);continue}if(e){this.infiniteLoopError(e.charCodeAt(0));break}}return t}infiniteLoopError(e){let t="Infinite loop on byte: "+e;if(this.options.silent)console.error(t);else throw new Error(t)}};var y=class{options;parser;constructor(e){this.options=e||T;}space(e){return ""}code({text:e,lang:t,escaped:n}){let s=(t||"").match(m.notSpaceStart)?.[0],r=e.replace(m.endingNewline,"")+`
`;return s?'<pre><code class="language-'+O(s)+'">'+(n?r:O(r,true))+`</code></pre>
`:"<pre><code>"+(n?r:O(r,true))+`</code></pre>
`}blockquote({tokens:e}){return `<blockquote>
${this.parser.parse(e)}</blockquote>
`}html({text:e}){return e}def(e){return ""}heading({tokens:e,depth:t}){return `<h${t}>${this.parser.parseInline(e)}</h${t}>
`}hr(e){return `<hr>
`}list(e){let t=e.ordered,n=e.start,s="";for(let o=0;o<e.items.length;o++){let u=e.items[o];s+=this.listitem(u);}let r=t?"ol":"ul",i=t&&n!==1?' start="'+n+'"':"";return "<"+r+i+`>
`+s+"</"+r+`>
`}listitem(e){return `<li>${this.parser.parse(e.tokens)}</li>
`}checkbox({checked:e}){return "<input "+(e?'checked="" ':"")+'disabled="" type="checkbox"> '}paragraph({tokens:e}){return `<p>${this.parser.parseInline(e)}</p>
`}table(e){let t="",n="";for(let r=0;r<e.header.length;r++)n+=this.tablecell(e.header[r]);t+=this.tablerow({text:n});let s="";for(let r=0;r<e.rows.length;r++){let i=e.rows[r];n="";for(let o=0;o<i.length;o++)n+=this.tablecell(i[o]);s+=this.tablerow({text:n});}return s&&(s=`<tbody>${s}</tbody>`),`<table>
<thead>
`+t+`</thead>
`+s+`</table>
`}tablerow({text:e}){return `<tr>
${e}</tr>
`}tablecell(e){let t=this.parser.parseInline(e.tokens),n=e.header?"th":"td";return (e.align?`<${n} align="${e.align}">`:`<${n}>`)+t+`</${n}>
`}strong({tokens:e}){return `<strong>${this.parser.parseInline(e)}</strong>`}em({tokens:e}){return `<em>${this.parser.parseInline(e)}</em>`}codespan({text:e}){return `<code>${O(e,true)}</code>`}br(e){return "<br>"}del({tokens:e}){return `<del>${this.parser.parseInline(e)}</del>`}link({href:e,title:t,tokens:n}){let s=this.parser.parseInline(n),r=V(e);if(r===null)return s;e=r;let i='<a href="'+e+'"';return t&&(i+=' title="'+O(t)+'"'),i+=">"+s+"</a>",i}image({href:e,title:t,text:n,tokens:s}){s&&(n=this.parser.parseInline(s,this.parser.textRenderer));let r=V(e);if(r===null)return O(n);e=r;let i=`<img src="${e}" alt="${O(n)}"`;return t&&(i+=` title="${O(t)}"`),i+=">",i}text(e){return "tokens"in e&&e.tokens?this.parser.parseInline(e.tokens):"escaped"in e&&e.escaped?e.text:O(e.text)}};var L=class{strong({text:e}){return e}em({text:e}){return e}codespan({text:e}){return e}del({text:e}){return e}html({text:e}){return e}text({text:e}){return e}link({text:e}){return ""+e}image({text:e}){return ""+e}br(){return ""}checkbox({raw:e}){return e}};var b=class l{options;renderer;textRenderer;constructor(e){this.options=e||T,this.options.renderer=this.options.renderer||new y,this.renderer=this.options.renderer,this.renderer.options=this.options,this.renderer.parser=this,this.textRenderer=new L;}static parse(e,t){return new l(t).parse(e)}static parseInline(e,t){return new l(t).parseInline(e)}parse(e){this.renderer.parser=this;let t="";for(let n=0;n<e.length;n++){let s=e[n];if(this.options.extensions?.renderers?.[s.type]){let i=s,o=this.options.extensions.renderers[i.type].call({parser:this},i);if(o!==false||!["space","hr","heading","code","table","blockquote","list","html","def","paragraph","text"].includes(i.type)){t+=o||"";continue}}let r=s;switch(r.type){case "space":{t+=this.renderer.space(r);break}case "hr":{t+=this.renderer.hr(r);break}case "heading":{t+=this.renderer.heading(r);break}case "code":{t+=this.renderer.code(r);break}case "table":{t+=this.renderer.table(r);break}case "blockquote":{t+=this.renderer.blockquote(r);break}case "list":{t+=this.renderer.list(r);break}case "checkbox":{t+=this.renderer.checkbox(r);break}case "html":{t+=this.renderer.html(r);break}case "def":{t+=this.renderer.def(r);break}case "paragraph":{t+=this.renderer.paragraph(r);break}case "text":{t+=this.renderer.text(r);break}default:{let i='Token with "'+r.type+'" type was not found.';if(this.options.silent)return console.error(i),"";throw new Error(i)}}}return t}parseInline(e,t=this.renderer){this.renderer.parser=this;let n="";for(let s=0;s<e.length;s++){let r=e[s];if(this.options.extensions?.renderers?.[r.type]){let o=this.options.extensions.renderers[r.type].call({parser:this},r);if(o!==false||!["escape","html","link","image","strong","em","codespan","br","del","text"].includes(r.type)){n+=o||"";continue}}let i=r;switch(i.type){case "escape":{n+=t.text(i);break}case "html":{n+=t.html(i);break}case "link":{n+=t.link(i);break}case "image":{n+=t.image(i);break}case "checkbox":{n+=t.checkbox(i);break}case "strong":{n+=t.strong(i);break}case "em":{n+=t.em(i);break}case "codespan":{n+=t.codespan(i);break}case "br":{n+=t.br(i);break}case "del":{n+=t.del(i);break}case "text":{n+=t.text(i);break}default:{let o='Token with "'+i.type+'" type was not found.';if(this.options.silent)return console.error(o),"";throw new Error(o)}}}return n}};var P=class{options;block;constructor(e){this.options=e||T;}static passThroughHooks=new Set(["preprocess","postprocess","processAllTokens","emStrongMask"]);static passThroughHooksRespectAsync=new Set(["preprocess","postprocess","processAllTokens"]);preprocess(e){return e}postprocess(e){return e}processAllTokens(e){return e}emStrongMask(e){return e}provideLexer(e=this.block){return e?x.lex:x.lexInline}provideParser(e=this.block){return e?b.parse:b.parseInline}};var q=class{defaults=M();options=this.setOptions;parse=this.parseMarkdown(true);parseInline=this.parseMarkdown(false);Parser=b;Renderer=y;TextRenderer=L;Lexer=x;Tokenizer=w;Hooks=P;constructor(...e){this.use(...e);}walkTokens(e,t){let n=[];for(let s of e)switch(n=n.concat(t.call(this,s)),s.type){case "table":{let r=s;for(let i of r.header)n=n.concat(this.walkTokens(i.tokens,t));for(let i of r.rows)for(let o of i)n=n.concat(this.walkTokens(o.tokens,t));break}case "list":{let r=s;n=n.concat(this.walkTokens(r.items,t));break}default:{let r=s;this.defaults.extensions?.childTokens?.[r.type]?this.defaults.extensions.childTokens[r.type].forEach(i=>{let o=r[i].flat(1/0);n=n.concat(this.walkTokens(o,t));}):r.tokens&&(n=n.concat(this.walkTokens(r.tokens,t)));}}return n}use(...e){let t=this.defaults.extensions||{renderers:{},childTokens:{}};return e.forEach(n=>{let s={...n};if(s.async=this.defaults.async||s.async||false,n.extensions&&(n.extensions.forEach(r=>{if(!r.name)throw new Error("extension name required");if("renderer"in r){let i=t.renderers[r.name];i?t.renderers[r.name]=function(...o){let u=r.renderer.apply(this,o);return u===false&&(u=i.apply(this,o)),u}:t.renderers[r.name]=r.renderer;}if("tokenizer"in r){if(!r.level||r.level!=="block"&&r.level!=="inline")throw new Error("extension level must be 'block' or 'inline'");let i=t[r.level];i?i.unshift(r.tokenizer):t[r.level]=[r.tokenizer],r.start&&(r.level==="block"?t.startBlock?t.startBlock.push(r.start):t.startBlock=[r.start]:r.level==="inline"&&(t.startInline?t.startInline.push(r.start):t.startInline=[r.start]));}"childTokens"in r&&r.childTokens&&(t.childTokens[r.name]=r.childTokens);}),s.extensions=t),n.renderer){let r=this.defaults.renderer||new y(this.defaults);for(let i in n.renderer){if(!(i in r))throw new Error(`renderer '${i}' does not exist`);if(["options","parser"].includes(i))continue;let o=i,u=n.renderer[o],a=r[o];r[o]=(...c)=>{let p=u.apply(r,c);return p===false&&(p=a.apply(r,c)),p||""};}s.renderer=r;}if(n.tokenizer){let r=this.defaults.tokenizer||new w(this.defaults);for(let i in n.tokenizer){if(!(i in r))throw new Error(`tokenizer '${i}' does not exist`);if(["options","rules","lexer"].includes(i))continue;let o=i,u=n.tokenizer[o],a=r[o];r[o]=(...c)=>{let p=u.apply(r,c);return p===false&&(p=a.apply(r,c)),p};}s.tokenizer=r;}if(n.hooks){let r=this.defaults.hooks||new P;for(let i in n.hooks){if(!(i in r))throw new Error(`hook '${i}' does not exist`);if(["options","block"].includes(i))continue;let o=i,u=n.hooks[o],a=r[o];P.passThroughHooks.has(i)?r[o]=c=>{if(this.defaults.async&&P.passThroughHooksRespectAsync.has(i))return (async()=>{let k=await u.call(r,c);return a.call(r,k)})();let p=u.call(r,c);return a.call(r,p)}:r[o]=(...c)=>{if(this.defaults.async)return (async()=>{let k=await u.apply(r,c);return k===false&&(k=await a.apply(r,c)),k})();let p=u.apply(r,c);return p===false&&(p=a.apply(r,c)),p};}s.hooks=r;}if(n.walkTokens){let r=this.defaults.walkTokens,i=n.walkTokens;s.walkTokens=function(o){let u=[];return u.push(i.call(this,o)),r&&(u=u.concat(r.call(this,o))),u};}this.defaults={...this.defaults,...s};}),this}setOptions(e){return this.defaults={...this.defaults,...e},this}lexer(e,t){return x.lex(e,t??this.defaults)}parser(e,t){return b.parse(e,t??this.defaults)}parseMarkdown(e){return (n,s)=>{let r={...s},i={...this.defaults,...r},o=this.onError(!!i.silent,!!i.async);if(this.defaults.async===true&&r.async===false)return o(new Error("marked(): The async option was set to true by an extension. Remove async: false from the parse options object to return a Promise."));if(typeof n>"u"||n===null)return o(new Error("marked(): input parameter is undefined or null"));if(typeof n!="string")return o(new Error("marked(): input parameter is of type "+Object.prototype.toString.call(n)+", string expected"));if(i.hooks&&(i.hooks.options=i,i.hooks.block=e),i.async)return (async()=>{let u=i.hooks?await i.hooks.preprocess(n):n,c=await(i.hooks?await i.hooks.provideLexer(e):e?x.lex:x.lexInline)(u,i),p=i.hooks?await i.hooks.processAllTokens(c):c;i.walkTokens&&await Promise.all(this.walkTokens(p,i.walkTokens));let h=await(i.hooks?await i.hooks.provideParser(e):e?b.parse:b.parseInline)(p,i);return i.hooks?await i.hooks.postprocess(h):h})().catch(o);try{i.hooks&&(n=i.hooks.preprocess(n));let a=(i.hooks?i.hooks.provideLexer(e):e?x.lex:x.lexInline)(n,i);i.hooks&&(a=i.hooks.processAllTokens(a)),i.walkTokens&&this.walkTokens(a,i.walkTokens);let p=(i.hooks?i.hooks.provideParser(e):e?b.parse:b.parseInline)(a,i);return i.hooks&&(p=i.hooks.postprocess(p)),p}catch(u){return o(u)}}}onError(e,t){return n=>{if(n.message+=`
Please report this to https://github.com/markedjs/marked.`,e){let s="<p>An error occurred:</p><pre>"+O(n.message+"",true)+"</pre>";return t?Promise.resolve(s):s}if(t)return Promise.reject(n);throw n}}};var z=new q;function g(l,e){return z.parse(l,e)}g.options=g.setOptions=function(l){return z.setOptions(l),g.defaults=z.defaults,N(g.defaults),g};g.getDefaults=M;g.defaults=T;g.use=function(...l){return z.use(...l),g.defaults=z.defaults,N(g.defaults),g};g.walkTokens=function(l,e){return z.walkTokens(l,e)};g.parseInline=z.parseInline;g.Parser=b;g.parser=b.parse;g.Renderer=y;g.TextRenderer=L;g.Lexer=x;g.lexer=x.lex;g.Tokenizer=w;g.Hooks=P;g.parse=g;g.options;g.setOptions;g.use;g.walkTokens;g.parseInline;b.parse;x.lex;

  /**
   * Represents a file.
   * @constructor
   * @param {string} url - The URL of the file.
   * @param {string} type - The type of file (i.e. posts, pages).
   * @param {object} layout - The layout templates of the file.
   */
  var File = /*#__PURE__*/function () {
    function File(url, type, layout, config) {
      _classCallCheck(this, File);
      this.url = type === 'SERVER' ? type + '/' + url : url;
      this.type = type;
      this.layout = layout;
      this.config = config;
      this.html = false;
      this.content;
      this.name;
      this.extension;
      this.title;
      this.excerpt;
      this.date;
      this.datetime;
      this.author;
      this.body;
      this.permalink;
      this.tags;
    }

    /**
     * Get file content.
     * @method
     * @async
     * @param {function} callback - Callback function.
     * @description
     * Get the file's HTML content and set the file object html
     * attribute to the file content.
     */
    return _createClass(File, [{
      key: "getContent",
      value: function getContent(callback) {
        var _this = this;
        get(this.url, function (success, error) {
          if (error) callback(success, error);
          _this.content = success;
          // check if the response returns a string instead
          // of an response object
          if (typeof _this.content === 'string') {
            callback(success, error);
          }
        });
      }

      /**
       * Parse front matter.
       * @method
       * @description
       * Overrides post attributes if front matter is available.
       */
    }, {
      key: "parseFrontMatter",
      value: function parseFrontMatter() {
        var yaml = this.content.split(this.config.frontMatterSeperator)[1];
        if (yaml) {
          var attributes = {};
          yaml.split(/\n/g).forEach(function (attributeStr) {
            var index = attributeStr.indexOf(':');
            if (index > -1) {
              var key = attributeStr.substring(0, index).trim();
              var val = attributeStr.substring(index + 1).trim();
              if (val) {
                attributes[key] = val;
              }
            }
          });
          extend(this, attributes);
        }
      }

      /**
       * Set list attributes.
       * @method
       * @description
       * Sets front matter attributes that are specified as list attributes to
       * an array by splitting the string by commas.
       */
    }, {
      key: "setListAttributes",
      value: function setListAttributes() {
        var _this2 = this;
        this.config.listAttributes.forEach(function (attribute) {
          if (Object.prototype.hasOwnProperty.call(_this2, attribute) && _this2[attribute]) {
            var val = _this2[attribute].trim();
            if (val.indexOf('[') === 0 && val.indexOf(']') === val.length - 1) {
              val = val.substring(1, val.length - 1);
            }
            _this2[attribute] = val.split(',').map(function (item) {
              return item.trim();
            });
          }
        });
      }

      /**
       * Sets filename.
       * @method
       */
    }, {
      key: "setFilename",
      value: function setFilename() {
        this.name = this.url.substr(this.url.lastIndexOf('/')).replace('/', '').replace(this.config.extension, '');
      }

      /**
       * Sets permalink.
       * @method
       */
    }, {
      key: "setPermalink",
      value: function setPermalink() {
        this.permalink = ['#', this.type, this.name].join('/');
      }

      /**
       * Set file date.
       * @method
       * @description
       * Check if file has date in front matter otherwise use the date
       * in the filename.
       */
    }, {
      key: "setDate",
      value: function setDate() {
        var dateRegEx = new RegExp(this.config.dateParser);
        if (this.date) {
          this.datetime = getDatetime(this.date);
          this.date = this.config.dateFormat(this.datetime);
        } else if (dateRegEx.test(this.url)) {
          this.date = dateRegEx.exec(this.url);
          this.datetime = getDatetime(this.date);
          this.date = this.config.dateFormat(this.datetime);
        }
      }

      /**
       * Set file body.
       * @method
       * @description
       * Sets the body of the file based on content after the front matter.
       */
    }, {
      key: "setBody",
      value: function setBody() {
        var html = this.content.split(this.config.frontMatterSeperator).splice(2).join(this.config.frontMatterSeperator);
        if (this.html) {
          this.body = html;
        } else {
          if (this.config.markdownEngine) {
            this.body = this.config.markdownEngine(html);
          } else {
            this.body = g.parse(html);
          }
        }
      }

      /**
       * Parse file content.
       * @method
       * @description
       * Sets all file attributes and content.
       */
    }, {
      key: "parseContent",
      value: function parseContent() {
        this.setFilename();
        this.setPermalink();
        this.parseFrontMatter();
        this.setListAttributes();
        this.setDate();
        this.setBody();
      }

      /**
       * Renders file.
       * @method
       * @async
       */
    }, {
      key: "render",
      value: function render() {
        return renderLayout(this.layout, this.config, this);
      }
    }]);
  }();

  /**
   * Represents a file collection.
   * @constructor
   * @param {string} type - The type of file collection (i.e. posts, pages).
   * @param {object} layout - The layouts of the file collection type.
   */
  var FileCollection = /*#__PURE__*/function () {
    function FileCollection(type, layout, config) {
      _classCallCheck(this, FileCollection);
      this.type = type;
      this.layout = layout;
      this.config = config;
      this.files = [];
      this[type] = this.files;
    }

    /**
     * Initialize file collection.
     * @method
     * @async
     * @param {function} callback - Callback function
     */
    return _createClass(FileCollection, [{
      key: "init",
      value: function init(callback) {
        var _this = this;
        this.getFiles(function (success, error) {
          if (error) handleMessage(messages['DIRECTORY_ERROR']);
          _this.loadFiles(function (success, error) {
            if (error) handleMessage(messages['GET_FILE_ERROR']);
            callback();
          });
        });
      }

      /**
       * Get file list URL.
       * @method
       * @param {string} type - Type of file collection.
       * @returns {string} URL of file list
       */
    }, {
      key: "getFileListUrl",
      value: function getFileListUrl(type, config) {
        return config.mode === 'GITHUB' ? getGithubUrl(type, config.github) : type;
      }

      /**
       * Get file URL.
       * @method
       * @param {object} file - File object.
       * @returns {string} File URL
       */
    }, {
      key: "getFileUrl",
      value: function getFileUrl(file, mode, type) {
        return mode === 'GITHUB' ? file['download_url'] : "".concat(type, "/").concat(getFilenameFromPath(file.getAttribute('href')));
      }

      /**
       * Get file elements.
       * @param {object} data - File directory or Github data.
       * @returns {array} File elements
       */
    }, {
      key: "getFileElements",
      value: function getFileElements(data) {
        var fileElements;

        // Github Mode
        if (this.config.mode === 'GITHUB') {
          fileElements = JSON.parse(data);
        }
        // Server Mode
        else {
          // convert the directory listing to a DOM element
          var listElement = document.createElement('div');
          listElement.innerHTML = data;
          // get the links in the directory listing
          fileElements = [].slice.call(listElement.getElementsByTagName('a'));
        }
        return fileElements;
      }

      /**
       * Get files from file listing and set to file collection.
       * @method
       * @async
       * @param {function} callback - Callback function
       */
    }, {
      key: "getFiles",
      value: function getFiles(callback) {
        var _this2 = this;
        get(this.getFileListUrl(this.type, this.config), function (success, error) {
          if (error) callback(success, error);
          // find the file elements that are valid files, exclude others
          _this2.getFileElements(success).forEach(function (file) {
            var fileUrl = _this2.getFileUrl(file, _this2.config.mode, _this2.type);
            if (isValidFile(fileUrl, _this2.config.extension)) {
              _this2.files.push(new File(fileUrl, _this2.type, _this2.layout.single, _this2.config));
            }
          });
          callback(success, error);
        });
      }

      /**
       * Load files and get file content.
       * @method
       * @async
       * @param {function} callback - Callback function
       */
    }, {
      key: "loadFiles",
      value: function loadFiles(callback) {
        var _this3 = this;
        var promises = [];
        if (this.files.length === 0) {
          callback(null, null);
          return;
        }
        // Load file content
        this.files.forEach(function (file, i) {
          file.getContent(function (success, error) {
            if (error) callback(success, error);
            promises.push(i);
            file.parseContent();
            // Execute after all content is loaded
            if (_this3.files.length == promises.length) {
              callback(success, error);
            }
          });
        });
      }

      /**
       * Search file collection by attribute.
       * @method
       * @param {string} attribute - Attribue in file to search.
       * @param {string} search - Search query.
       * @returns {object} File object
       */
    }, {
      key: "search",
      value: function search(attribute, _search) {
        this[this.type] = this.files.filter(function (file) {
          var attr = (file[attribute] || '').toLowerCase().trim();
          return attr.indexOf(_search.toLowerCase().trim()) >= 0;
        });
      }

      /**
       * Reset file collection files.
       * @method
       */
    }, {
      key: "resetSearch",
      value: function resetSearch() {
        this[this.type] = this.files;
      }

      /**
       * Get files by tag.
       * @method
       * @param {string} query - Search query.
       * @returns {array} Files array
       */
    }, {
      key: "getByTag",
      value: function getByTag(query) {
        this[this.type] = this.files.filter(function (file) {
          if (query && file.tags) {
            return file.tags.some(function (tag) {
              return tag === query;
            });
          }
        });
      }

      /**
       * Get file by permalink.
       * @method
       * @param {string} permalink - Permalink to search.
       * @returns {object} File object.
       */
    }, {
      key: "getFileByPermalink",
      value: function getFileByPermalink(permalink) {
        return this.files.filter(function (file) {
          return file.permalink === permalink;
        })[0];
      }

      /**
       * Renders file collection.
       * @method
       * @async
       * @returns {string} Rendered layout
       */
    }, {
      key: "render",
      value: function render() {
        return renderLayout(this.layout.list, this.config, this);
      }
    }]);
  }();

  /**
   * Represents a CMS instance
   * @constructor
   * @param {object} options - Configuration options.
   */
  var CMS = /*#__PURE__*/function () {
    function CMS(view, options) {
      _classCallCheck(this, CMS);
      this.ready = false;
      this.collections = {};
      this.filteredCollections = {};
      this.state;
      this.view = view;
      this.config = Object.assign({}, defaults, options);
      this.init();
    }

    /**
     * Init
     * @method
     * @description
     * Initializes the application based on the configuration. Sets up up config object,
     * hash change event listener for router, and loads the content.
     */
    return _createClass(CMS, [{
      key: "init",
      value: function init() {
        var _this = this;
        // create message container element if debug mode is enabled
        if (this.config.debug) {
          createMessageContainer(this.config.messageClassName);
        }
        if (this.config.elementId) {
          // setup container
          this.config.container = document.getElementById(this.config.elementId);
          if (this.config.container) {
            // setup file collections
            this.initFileCollections(function () {
              // check for hash changes
              _this.view.addEventListener('hashchange', _this.route.bind(_this), false);
              // start router by manually triggering hash change
              _this.view.dispatchEvent(new HashChangeEvent('hashchange'));
              // register plugins and run onload events
              _this.ready = true;
              _this.registerPlugins();
              _this.config.onload();
            });
          } else {
            handleMessage(this.config.debug, messages['ELEMENT_ID_ERROR']);
          }
        } else {
          handleMessage(this.config.debug, messages['ELEMENT_ID_ERROR']);
        }
      }

      /**
       * Initialize file collections
       * @method
       * @async
       */
    }, {
      key: "initFileCollections",
      value: function initFileCollections(callback) {
        var _this2 = this;
        var promises = [];
        var types = [];

        // setup collections and routes
        this.config.types.forEach(function (type) {
          _this2.collections[type.name] = new FileCollection(type.name, type.layout, _this2.config);
          types.push(type.name);
        });

        // init collections
        types.forEach(function (type, i) {
          _this2.collections[type].init(function () {
            promises.push(i);
            // reverse order to display newest posts first for post types
            if (type.indexOf('post') === 0) {
              _this2.collections[type][type].reverse();
            }
            // Execute after all content is loaded
            if (types.length == promises.length) {
              callback();
            }
          });
        });
      }
    }, {
      key: "route",
      value: function route() {
        var paths = getPathsWithoutParameters();
        var type = paths[0];
        var filename = paths[1];
        var collection = this.collections[type];
        var query = getParameterByName('query') || '';
        var tag = getParameterByName('tag') || '';
        this.state = window.location.hash.substr(1);

        // Default view
        if (!type) {
          window.location = ['#', this.config.defaultView].join('/');
        }
        // List and single views
        else {
          if (filename && collection) {
            // Single view
            var permalink = ['#', type, filename.trim()].join('/');
            var file = collection.getFileByPermalink(permalink);
            if (file) {
              file.render();
            } else {
              renderLayout(this.config.errorLayout, this.config, {});
            }
          } else if (collection) {
            // List view
            if (query) {
              // Check for queries
              collection.search('title', query);
            } else if (tag) {
              // Check for tags
              collection.getByTag(tag);
            } else {
              // Reset search
              collection.resetSearch();
            }
            collection.render();
          } else {
            // Error view
            renderLayout(this.config.errorLayout, this.config, {});
          }
        }
        // onroute event
        this.config.onroute();
      }

      /**
       * Register plugins.
       * @method
       * @description
       * Set up plugins based on user configuration.
       */
    }, {
      key: "registerPlugins",
      value: function registerPlugins() {
        var _this3 = this;
        this.config.plugins.forEach(function (plugin) {
          var name = getFunctionName(plugin);
          if (!_this3[name]) {
            _this3[name] = plugin;
          }
        });
      }

      /**
       * Sort method for file collections.
       * @method
       * @param {string} type - Type of file collection.
       * @param {function} sort - Sorting function.
       */
    }, {
      key: "sort",
      value: function sort(type, _sort) {
        if (this.ready) {
          this.collections[type][type].sort(_sort);
          this.collections[type].render();
        } else {
          handleMessage(messages['NOT_READY_WARNING']);
        }
      }

      /**
       * Search method for file collections.
       * @method
       * @param {string} type - Type of file collection.
       * @param {string} attribute - File attribute to search.
       * @param {string} search - Search query.
       */
    }, {
      key: "search",
      value: function search(type, attribute, _search) {
        if (this.ready) {
          this.collections[type].search(attribute, _search);
          this.collections[type].render();
        } else {
          handleMessage(messages['NOT_READY_WARNING']);
        }
      }
    }]);
  }();

  /**
   * CMS.js v2.0.0
   * Copyright 2018 Chris Diana
   * https://chrisdiana.github.io/cms.js
   * Free to use under the MIT license.
   * http://www.opensource.org/licenses/mit-license.php
   */
  var main = (function (options) {
    return new CMS(window, options);
  });

  return main;

})();
