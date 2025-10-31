import { nodeResolve } from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import { babel } from '@rollup/plugin-babel';
import { terser } from '@rollup/plugin-terser';
import livereload from 'rollup-plugin-livereload';
import { readFileSync } from 'fs';

const packageJson = JSON.parse(readFileSync('./package.json', 'utf8'));
const { name, version, license, author, homepage } = packageJson;
const production = !process.env.ROLLUP_WATCH;
const banner = `/*! ${name} v${version} | ${license} (c) ${new Date().getFullYear()} ${author.name} | ${homepage} */`;

const outputs = [];

if (production) {
  outputs.push({
    file: 'dist/cms.min.js',
    name: 'CMS',
    format: 'iife',
    banner: banner,
    plugins: [
      terser({ 
        format: { 
          comments: /^!/ 
        }
      })
    ]
  });
  outputs.push({
    file: 'dist/cms.js',
    name: 'CMS',
    format: 'iife',
    banner: banner
  });
  outputs.push({
    file: 'dist/cms.es.js',
    name: 'CMS',
    format: 'es',
    banner: banner
  });
} else {
  outputs.push({
    file: 'dist/cms.js',
    name: 'CMS',
    format: 'iife',
    banner: banner
  });
  // También generar para examples durante desarrollo
  outputs.push({
    file: 'examples/js/cms.js',
    name: 'CMS',
    format: 'iife',
    banner: banner
  });
}

export default {
  input: 'src/main.js',
  external: ['CMS'],
  output: outputs,
  plugins: [
    nodeResolve({
      browser: true,
      preferBuiltins: false
    }),
    commonjs(),
    babel({ 
      exclude: 'node_modules/**',
      babelHelpers: 'bundled',
      presets: ['@babel/preset-env']
    }),
    !production && livereload({
      watch: ['dist', '.']
    })
  ]
};
