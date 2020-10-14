const chalk = require('chalk');
const { parse } = require('./src/json.pjs');
const { psw, removeLinebreak, verboseLog } = require('./src/utils');
const fixer = require('./src/fixer');

let fixRounds = 0;
let roundThreshold = 20;

const setFixThreshold = (data) => {
  const lineCount = data.split('\n').length;
  roundThreshold = Math.max(data.length / lineCount, lineCount);
};

const doubleCheck = (data, options = {}) => {
  /* eslint-disable no-console */
  const verbose = options.verbose;
  try {
    const res = parse(data);
    psw(`\n${chalk.cyan('The JSON data was fixed!')}`);
    if (res) {
      return options.parse ? res : data;
    }
  } catch (err) {
    if (verbose) {
      psw('Nearly fixed data:');
      data.split('\n').forEach((l, i) => psw(`${chalk.yellow(i)} ${l}`));
    }
    // eslint-disable-next-line no-use-before-define
    if (fixRounds < roundThreshold) return fixJson(err, data, options);
    console.error(chalk.red("There's still an error!"));
    throw new Error(err.message);
  }
  /* eslint-enable no-console */
};

const extraChar = (err) => err.expected[0].type === 'other' && ['}', ']'].includes(err.found);

const trailingChar = (err) => {
  const literal = err.expected[0].type === 'literal' && err.expected[0].text !== ':';
  return ['.', ',', 'x', 'b', 'o'].includes(err.found) && literal;
};

const missingChar = (err) => err.expected[0].text === ',' && ['"', '[', '{'].includes(err.found);

const singleQuotes = (err) => err.found === "'";

const missingQuotes = (err) =>
  /\w/.test(err.found) && err.expected.find((el) => el.description === 'string');

const notSquare = (err) => err.found === ':' && [',', ']'].includes(err.expected[0].text);

const notCurly = (err) => err.found === ',' && err.expected[0].text === ':';

const comment = (err) => err.found === '/';

const ops = (err) => ['+', '-', '*', '/', '>', '<', '~', '|', '&', '^'].includes(err.found);

const extraBrackets = (err) => err.found === '}';

const specialChar = (err) => err.found === '"';

const runFixer = ({ verbose, lines, start, err }) => {
  /* eslint-disable security/detect-object-injection */
  let fixedData = [...lines];
  const targetLine = start.line - 2;

  if (extraChar(err)) {
    fixedData = fixer.fixExtraChar({ fixedData, verbose, targetLine });
  } else if (trailingChar(err)) {
    fixedData = fixer.fixTrailingChar({ start, fixedData, verbose });
  } else if (missingChar(err)) {
    if (verbose) psw(chalk.magenta('Missing character'));
    const brokenLine = removeLinebreak(lines[targetLine]);
    fixedData[targetLine] = `${brokenLine},`;
  } else if (singleQuotes(err)) {
    fixedData = fixer.fixSingleQuotes({ start, fixedData, verbose });
  } else if (missingQuotes(err)) {
    fixedData = fixer.fixMissingQuotes({ start, fixedData, verbose });
  } else if (notSquare(err)) {
    fixedData = fixer.fixSquareBrackets({ start, fixedData, verbose, targetLine });
  } else if (notCurly(err)) {
    fixedData = fixer.fixCurlyBrackets({ fixedData, verbose, targetLine });
  } else if (comment(err)) {
    fixedData = fixer.fixComment({ start, fixedData, verbose });
  } else if (ops(err)) {
    fixedData = fixer.fixOpConcat({ start, fixedData, verbose });
  } else if (extraBrackets(err)) {
    fixedData = fixer.fixExtraCurlyBrackets({ start, fixedData, verbose });
  } else if (specialChar(err)) {
    fixedData = fixer.fixSpecialChar({ start, fixedData, verbose });
  } else throw new Error(`Unsupported issue: ${err.message} (please open an issue at the repo)`);
  return fixedData;
};

/*eslint-disable no-console */
const fixJson = (err, data, options) => {
  ++fixRounds;
  const lines = data.split('\n');
  const verbose = options.verbose;
  verboseLog({ verbose, lines, err });

  const start = err.location.start;
  const fixedData = runFixer({ verbose, lines, start, err });

  return doubleCheck(fixedData.join('\n'), options);
};
/*eslint-enable no-console */

const fixingTime = ({ data, err, optionsCopy }) => {
  fixRounds = 0;
  setFixThreshold(data);
  return {
    data: fixJson(err, data, optionsCopy),
    changed: true
  };
};

/**
 * @param {string} data JSON string data to check (and fix).
 * @param {{verbose:boolean, parse:boolean}} options configuration object which specifies verbosity and whether the object should be parsed or returned as fixed string
 * @returns {{data: (Object|string|Array), changed: boolean}} Result
 */
const checkJson = (data, options) => {
  //inspired by https://jsontuneup.com/
  let optionsCopy;
  if (!options || typeof options === 'boolean') {
    optionsCopy = {};
    optionsCopy.verbose = options;
  } else {
    optionsCopy = JSON.parse(JSON.stringify(options));
  }

  if (optionsCopy.parse === undefined || optionsCopy.parse === null) {
    optionsCopy.parse = true;
  }

  try {
    const res = parse(data);
    if (res) {
      return {
        data: optionsCopy.parse ? res : data,
        changed: false
      };
    }
  } catch (err) {
    return fixingTime({ data, err, optionsCopy });
  }
};

module.exports = checkJson;
