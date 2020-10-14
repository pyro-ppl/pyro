const { removeLinebreak, replaceChar, curlyBracesIncluded } = require('../utils');

test('No line breaks', () => {
  expect(removeLinebreak(`a\nnew\nline`)).toEqual('anewline');
});

test('Spaces there', () => {
  expect(removeLinebreak('not the same ')).toEqual('not the same ');
});

test('Char replaced', () => {
  expect(replaceChar('Hi$mate', 2, ' ')).toEqual('Hi mate');
});

test('Char !not replaced', () => {
  expect(replaceChar('Hi$mate', -1, ' ')).toEqual(' Hi$mate');
});

describe('Inlined braces', () => {
  it('{ ... }', () => {
    expect(curlyBracesIncluded('{ a: "thing }')).toBeTruthy();
  });

  it('[ ... ]', () => {
    expect(curlyBracesIncluded('[ 0, 1 ]')).toBeFalsy();
  });
});

describe('Outlined braces', () => {
  it('{ ... }', () => {
    expect(curlyBracesIncluded('a: "thing\n')).toBeFalsy();
  });

  it('[ ... ]', () => {
    expect(curlyBracesIncluded('0, 1')).toBeFalsy();
  });
});
