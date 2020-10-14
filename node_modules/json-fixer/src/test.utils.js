const { readFileSync } = require('fs');
const jf = require('../');

const exam = ({ sampleName, expectedOutput, fixerOptions = {}, expectedChange = false } = {}) => {
  // eslint-disable-next-line security/detect-non-literal-fs-filename
  const json = readFileSync(`./test/samples/${sampleName}.json`, 'utf-8');
  const { data, changed } = jf(json, fixerOptions);
  expect(changed).toEqual(expectedChange);
  expect(data).toEqual(expectedOutput);
};

module.exports = { exam };
