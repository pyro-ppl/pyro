const chalk = require('chalk');

const psw = (data) => process.stdout.write(`${data}\n`);

const removeLinebreak = (line) => line.replace(/[\n\r]/g, '');

const replaceChar = (str, idx, chr) => str.substring(0, idx) + chr + str.substring(idx + 1);

const verboseLog = ({ verbose = false, lines = [], err }) => {
  if (!verbose) return;
  psw('Data:');
  lines.forEach((l, i) => psw(`${chalk.yellow(i)} ${l}`));
  psw(chalk.red('err='));
  console.dir(err);
};

const curlyBracesIncluded = (line) => {
  const l = line.trim();
  return l.startsWith('{') && l.endsWith('}');
};

module.exports = { psw, removeLinebreak, replaceChar, verboseLog, curlyBracesIncluded };
