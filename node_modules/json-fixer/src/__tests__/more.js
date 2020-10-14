const { readFileSync } = require('fs');
const jf = require('../..');

describe('returns the json as fixed string', () => {
  it('normal file', () => {
    const json = readFileSync('./test/samples/normal.json', 'utf-8');
    const { data } = jf(json, { parse: false });
    expect(typeof data).toBe('string');
    expect(typeof JSON.parse(data)).toBe('object');
  });
});

test('Unsupported error', () => {
  const json = readFileSync('./test/samples/quoteInQuotes.json', 'utf-8');
  expect(() => jf(json)).toThrowError(
    'Unsupported issue: Expected "," or "}" but "M" found. (please open an issue at the repo)'
  );
});
