const chalk = require('chalk');
const { psw, removeLinebreak, replaceChar, curlyBracesIncluded } = require('./utils');
const { quotify, numberify, baseNumify } = require('./transform');
const { parse } = require('./json.pjs');

const fixExtraChar = ({ fixedData, verbose, targetLine }) => {
  /* eslint-disable security/detect-object-injection */
  if (verbose) psw(chalk.magenta('Extra character'));
  if (fixedData[targetLine] === '') --targetLine;
  const brokenLine = removeLinebreak(fixedData[targetLine]);

  let fixedLine = brokenLine.trimEnd();
  fixedLine = fixedLine.substr(0, fixedLine.length - 1);
  fixedData[targetLine] = fixedLine;
  return fixedData;
};

const fixSingleQuotes = ({ start, fixedData, verbose }) => {
  if (verbose) psw(chalk.magenta('Single quotes'));
  const targetLine = start.line - 1;
  const brokenLine = removeLinebreak(fixedData[targetLine]);
  const fixedLine = brokenLine.replace(/(":\s*)'(.*?)'/g, '$1"$2"');
  fixedData[targetLine] = fixedLine;
  return fixedData;
};

const fixTrailingChar = ({ start, fixedData, verbose }) => {
  if (verbose) psw(chalk.magenta('Trailing character'));
  const targetLine = start.line - 1;
  const brokenLine = removeLinebreak(fixedData[targetLine]);
  const fixedLine = brokenLine.replace(/(":\s*)[.,](\d*)/g, '$10.$2');
  const unquotedWord = /(":\s*)(\S*)/g.exec(fixedLine);
  // if (unquotedWord === null) throw new Error('Unquoted word expected!');
  const NN = Number.isNaN(Number(unquotedWord[2]));
  if (NN && !/([xbo][0-9a-fA-F]+)/.test(unquotedWord[2])) {
    return quotify({ fixedData, targetLine, fixedLine, verbose });
  }
  if (!NN && !/\0([xbo][0-9a-fA-F]+)/.test(unquotedWord[2])) {
    return numberify({ fixedData, targetLine, fixedLine, unquotedWord, verbose });
  }
  let baseNumber = fixedLine.replace(/(":\s*)([xbo][0-9a-fA-F]*)/g, '$1"0$2"');
  if (baseNumber !== fixedLine) {
    baseNumber = baseNumify({ baseNumber, verbose });
  }

  fixedData[targetLine] = baseNumber;
  return fixedData;
};

const fixMissingQuotes = ({ start, fixedData, verbose }) => {
  /* eslint-disable security/detect-object-injection */
  if (verbose) psw(chalk.magenta('Missing quotes'));
  const targetLine = start.line - 1;
  let brokenLine = removeLinebreak(fixedData[targetLine]);
  const seCurlyBraces = curlyBracesIncluded(brokenLine);
  if (seCurlyBraces) {
    brokenLine = brokenLine.substring(1, brokenLine.length - 1);
  }
  const NO_RH_QUOTES = /(":\s*)([^,{}[\]]+)/;
  const NO_LH_QUOTES = /(^[^"][\S\s]*)(:\s*["\w{[])/;
  const RH = NO_RH_QUOTES.test(brokenLine);
  let fixedLine = RH ? brokenLine.replace(NO_RH_QUOTES, '$1"$2"') : brokenLine;
  const leftSpace = fixedLine.match(/^(\s+)/);
  fixedLine = fixedLine.trimStart();
  if (NO_LH_QUOTES.test(fixedLine)) {
    const firstColon = fixedLine.indexOf(':');
    const leftHand = fixedLine.substring(0, firstColon);
    fixedLine = `"${leftHand}"${fixedLine.substring(firstColon)}`;
  }
  fixedData[targetLine] = `${leftSpace === null ? '' : leftSpace[0]}${fixedLine}`;
  if (seCurlyBraces) {
    fixedData[targetLine] = `{${fixedData[targetLine]}}`;
  }

  return fixedData;
};

const fixSquareBrackets = ({ start, fixedData, verbose, targetLine }) => {
  /* eslint-disable security/detect-object-injection */
  if (verbose) psw(chalk.magenta('Square brackets instead of curly ones'));
  const lineToChange = fixedData[targetLine].includes('[')
    ? fixedData[targetLine]
    : fixedData[++targetLine];
  const brokenLine = removeLinebreak(lineToChange);
  const fixedLine = replaceChar(brokenLine, start.column - 1, '{');
  fixedData[targetLine] = fixedLine;

  try {
    parse(fixedData.join('\n'));
  } catch (e) {
    targetLine = e.location.start.line - 1;
    const newLine = removeLinebreak(fixedData[targetLine]).replace(']', '}');
    fixedData[targetLine] = newLine;
  }
  return fixedData;
};

const fixCurlyBrackets = ({ fixedData, verbose, targetLine }) => {
  if (verbose) psw(chalk.magenta('Curly brackets instead of square ones'));
  const brokenLine = removeLinebreak(
    fixedData[targetLine].includes('{') ? fixedData[targetLine] : fixedData[++targetLine]
  );
  const fixedLine = replaceChar(brokenLine, brokenLine.indexOf('{'), '[');
  fixedData[targetLine] = fixedLine;

  try {
    parse(fixedData.join('\n'));
  } catch (e) {
    targetLine = e.location.start.line - 1;
    const newLine = removeLinebreak(fixedData[targetLine]).replace('}', ']');
    fixedData[targetLine] = newLine;
  }

  return fixedData;
};

const fixMultilineComment = ({ fixedData, targetLine }) => {
  let end = targetLine + 1;
  while (end <= fixedData.length && !fixedData[end].includes('*/')) ++end;
  for (let i = targetLine + 1; i <= end; ++i) fixedData[i] = '#RM';
  fixedData[targetLine] = fixedData[targetLine].replace(/\s*\/\*+.*/g, '#RM');
  return fixedData.filter((l) => l !== '#RM');
};

const fixComment = ({ start, fixedData, verbose }) => {
  if (verbose) psw(chalk.magenta('Comment'));
  const targetLine = start.line - 1;
  const brokenLine = removeLinebreak(fixedData[targetLine]);
  const fixedLine = brokenLine.replace(/(\s*)(\/\/.*|\/\*+.*?\*+\/)/g, '');
  if (fixedLine.includes('/*')) {
    return fixMultilineComment({ fixedData, targetLine });
  }
  fixedData[targetLine] = fixedLine;
  return fixedData;
};

const fixOpConcat = ({ start, fixedData, verbose }) => {
  if (verbose) psw(chalk.magenta('Operations/Concatenations'));
  psw(
    chalk.yellow(
      'Please note: calculations made here may not be entirely correct on complex operations'
    )
  );
  const targetLine = start.line - 1;
  const brokenLine = removeLinebreak(fixedData[targetLine]);
  const fixedLine = brokenLine
    /* eslint-disable no-eval, security/detect-eval-with-expression */
    .replace(
      /(\d+)\s*([+\-*/%&|^><]|\*\*|>{2,3}|<<|[=!><]=|[=!]==)\s*(\d+)\s*([+\-*/%&|^><]|\*\*|>{2,3}|<<|[=!><]=|[=!]==)*\s*(\d+)*/g,
      (eq) => eval(eq)
    )
    .replace(/[~!+-]\(?(\d+)\)?/g, (eq) => eval(eq))
    .replace(/(":\s*)"(.*?)"\s*\+\s*"(.*?)"/g, '$1"$2$3"');
  /* eslint-enable no-eval */
  fixedData[targetLine] = fixedLine;
  return fixedData;
};

const fixExtraCurlyBrackets = ({ start, fixedData, verbose }) => {
  if (verbose) psw(chalk.magenta('Extra curly brackets'));

  const targetLine = start.line - 1;
  const fullData = fixedData.join('\n');
  let fixedLine = removeLinebreak(fixedData[targetLine]);

  const data = fullData.split('');
  const openingCount = data.filter((c) => c === '{').length;
  const closingCount = data.filter((c) => c === '}').length;
  const bracketDiff = closingCount - openingCount;

  for (let i = 0; i < bracketDiff; i++) {
    const index = fixedLine.lastIndexOf('}');
    fixedLine = fixedLine.slice(0, index) + fixedLine.slice(index + 1);
  }

  fixedData[targetLine] = fixedLine;
  return fixedData;
};

const fixSpecialChar = ({ start, fixedData, verbose }) => {
  if (verbose) psw(chalk.magenta('Special character'));
  const targetLine = start.line - 1;
  const brokenLine = fixedData[targetLine];

  let fixedLine = brokenLine
    .replace(/\f/g, '\\f')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');

  if (brokenLine.endsWith('"') && brokenLine[start.column] === undefined) {
    if (verbose) psw(chalk.magenta('New line'));
    const removedIndex = targetLine + 1;
    const continuation = fixedData[removedIndex];
    fixedLine = `${brokenLine}\\n${continuation}`;
    fixedData.splice(removedIndex, 1);
  }

  fixedData[targetLine] = fixedLine;
  return fixedData;
};

module.exports = {
  fixExtraChar,
  fixSingleQuotes,
  fixTrailingChar,
  fixMissingQuotes,
  fixSquareBrackets,
  fixCurlyBrackets,
  fixComment,
  fixOpConcat,
  fixExtraCurlyBrackets,
  fixSpecialChar
};
