/* eslint-disable security/detect-object-injection */
const { psw } = require('./utils');
const chalk = require('chalk');

const quotify = ({ fixedData, targetLine, fixedLine, verbose }) => {
  if (verbose) psw(chalk.magenta('Adding quotes...'));
  fixedData[targetLine] = fixedLine.replace(/(":\s*)(\S*)/g, '$1"$2"');
  return fixedData;
};

const numberify = ({ fixedData, targetLine, fixedLine, unquotedWord, verbose }) => {
  if (verbose) {
    psw(
      chalk.cyan(
        "Found a non base-10 number and since JSON doesn't support those numbers types. I will turn it into a base-10 number to keep the structure intact"
      )
    );
  }
  fixedData[targetLine] = fixedLine.replace(unquotedWord[2], Number(unquotedWord[2]));
  return fixedData;
};

const baseNumify = ({ baseNumber, verbose }) => {
  if (verbose) {
    psw(
      chalk.cyan(
        "Found a non base-10 number and since JSON doesn't support those numbers types. I will turn it into a base-10 number to keep the structure intact"
      )
    );
  }
  return baseNumber.replace(/"(0[xbo][0-9a-fA-F]*)"/g, (_, num) => Number(num)); //base-(16|2|8) -> base-10
};

module.exports = {
  quotify,
  numberify,
  baseNumify
};
