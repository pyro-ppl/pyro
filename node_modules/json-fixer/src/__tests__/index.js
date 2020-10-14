const { exam } = require('../test.utils');

const shouldHaveChanged = (sampleName, expectedOutput, fixerOptions = {}) => {
  exam({ sampleName, expectedOutput, fixerOptions, expectedChange: true });
};

it('fix single quotes', () => {
  shouldHaveChanged('singleQuote', {
    name: 'sample #1',
    type: 'JSON',
    error: 'single quote',
    version: '1'
  });
});

describe('fix missing quotes', () => {
  it('RHS: one word', () => {
    shouldHaveChanged('noQuotes', {
      name: 'sample #10',
      type: 'JSON',
      error: 'missing quotes',
      version: 'one'
    });
  });

  it('RHS: one word (verbose)', () => {
    shouldHaveChanged(
      'noQuotes',
      {
        name: 'sample #10',
        type: 'JSON',
        error: 'missing quotes',
        version: 'one'
      },
      { verbose: true }
    );
  });

  it('RHS: several words', () => {
    shouldHaveChanged('missingQuotes', {
      name: 'sample #11',
      type: 'JSON',
      error: 'missing quotes',
      version: 'a string'
    });
  });

  it('LHS: one word', () => {
    shouldHaveChanged('noLHQuotes', {
      name: 'sample #13',
      type: 'JSON',
      error: 'missing quotes',
      version: 'a string'
    });
  });

  it('LHS: 2 chars', () => {
    shouldHaveChanged('lefty2', {
      ix: 1
    });
  });

  it('LHS: 1 char', () => {
    shouldHaveChanged('lefty1', {
      t: 42
    });
  });

  it('LHS: not an octet', () => {
    shouldHaveChanged('leftyO', {
      o: 1
    });
  });

  it('LHS: one word (verbose)', () => {
    shouldHaveChanged(
      'noLHQuotes',
      {
        name: 'sample #13',
        type: 'JSON',
        error: 'missing quotes',
        version: 'a string'
      },
      { verbose: true }
    );
  });

  it('LHS: several words', () => {
    shouldHaveChanged('missingLHQuotes', {
      name: 'sample #14',
      type: 'JSON',
      error: 'missing quotes',
      'long content': 'a string'
    });
  });

  it('LHS: complicated RHS', () => {
    shouldHaveChanged('issue31', {
      something: 'string:string'
    });
  });

  it('Both sides', () => {
    shouldHaveChanged('doublyMissingQuotes', {
      field: 'value'
    });
  });

  it('Both sides (minified)', () => {
    shouldHaveChanged('doublyMissingQuotesMin', {
      field: 'value'
    });
  });
});

describe('fix trailing characters', () => {
  it('dots', () => {
    shouldHaveChanged('trailingDot', {
      name: 'sample #3',
      type: 'JSON',
      error: 'trailing dot',
      version: 0.3
    });
  });

  it('commas', () => {
    shouldHaveChanged('trailingComma', {
      name: 'sample #6',
      type: 'JSON',
      error: 'trailing comma',
      version: 0.6
    });
  });

  it('chars', () => {
    shouldHaveChanged('trailingChar', [
      {
        test1: '1',
        test2: {
          a: 'b',
          c: {}
        }
      }
    ]);
  });

  it('hex\'s "x"', () => {
    shouldHaveChanged('x', {
      name: 'sample #7',
      type: 'JSON',
      error: 'trailing x',
      version: 0x7
    });
  });

  it('hex\'s "x" (verbose)', () => {
    shouldHaveChanged(
      'x',
      {
        name: 'sample #7',
        type: 'JSON',
        error: 'trailing x',
        version: 0x7
      },
      { verbose: true }
    );
  });

  it('hex\'s "0x"', () => {
    shouldHaveChanged('hex', {
      name: 'sample #22',
      type: 'JSON',
      error: 'hex number',
      version: 0x16
    });
  });

  it('hex\'s "0x" (verbose)', () => {
    shouldHaveChanged(
      'hex',
      {
        name: 'sample #22',
        type: 'JSON',
        error: 'hex number',
        version: 0x16
      },
      { verbose: true }
    );
  });

  it('binary\'s "b"', () => {
    shouldHaveChanged('b', {
      name: 'sample #8',
      type: 'JSON',
      error: 'trailing b',
      version: 0b1000
    });
  });

  it('binary\'s "0b"', () => {
    shouldHaveChanged('bin', {
      name: 'sample #23',
      type: 'JSON',
      error: 'binary number',
      version: 0b10111
    });
  });

  it('octal\'s "o"', () => {
    shouldHaveChanged('o', {
      name: 'sample #9',
      type: 'JSON',
      error: 'trailing o',
      version: 0o11
    });
  });

  it('octal\'s "0o"', () => {
    shouldHaveChanged('oct', {
      name: 'sample #24',
      type: 'JSON',
      error: 'octal number',
      version: 0o30
    });
  });
});

it('fix extra characters', () => {
  shouldHaveChanged('extraChar', {
    name: 'sample #4',
    type: 'JSON',
    error: 'trailing error',
    version: 4
  });
});

it('fix missing commas', () => {
  shouldHaveChanged('missing', {
    name: 'sample #5',
    type: 'JSON',
    error: 'missing comma',
    version: 5
  });
});

describe('fix wrong brackets', () => {
  it('square brackets', () => {
    shouldHaveChanged('notSquare', {
      name: 'sample #12',
      error: 'wrong brackets',
      info: {
        type: 'JSON',
        version: 12
      }
    });
  });

  it('square brackets (verbose)', () => {
    shouldHaveChanged(
      'notSquare',
      {
        name: 'sample #12',
        error: 'wrong brackets',
        info: {
          type: 'JSON',
          version: 12
        }
      },
      { verbose: true }
    );
  });

  it('curly brackets', () => {
    shouldHaveChanged('notCurly', {
      name: 'sample #15',
      error: 'wrong brackets',
      info: ['one', 'two']
    });
  });

  it('curly brackets (verbose)', () => {
    shouldHaveChanged(
      'notCurly',
      {
        name: 'sample #15',
        error: 'wrong brackets',
        info: ['one', 'two']
      },
      { verbose: true }
    );
  });

  it('extra brackets', () => {
    shouldHaveChanged('extraBrackets', {
      error: 'extra brackets'
    });
  });
});

describe('comments', () => {
  it('inline line', () => {
    shouldHaveChanged('comment', {
      name: 'sample #16',
      type: 'JSON',
      error: 'comment',
      version: '0x10'
    });
  });

  it('single line', () => {
    shouldHaveChanged('smComment', {
      name: 'sample #17',
      type: 'JSON',
      error: 'multi-comment',
      version: '0x10'
    });
  });

  it('multi line', () => {
    shouldHaveChanged('multiComment', {
      name: 'sample #18',
      type: 'JSON',
      error: 'multi-comment',
      version: 18
    });
  });
});

describe('fix operations', () => {
  it('simple', () => {
    shouldHaveChanged('ops', {
      name: 'sample #20',
      type: 'JSON',
      error: 'operations',
      version: 20
    });
  });

  it('unary', () => {
    shouldHaveChanged('monOps', {
      name: 'sample #26',
      type: 'JSON',
      error: 'unary operations',
      version: -7
    });
  });

  it('multi', () => {
    shouldHaveChanged('multiOps', {
      name: 'sample #27',
      type: 'JSON',
      error: 'multi operations',
      version: 7
    });
  });
});

describe('fix concatenations', () => {
  it('simple', () => {
    shouldHaveChanged('concat', {
      name: 'sample #25',
      type: 'JSON',
      error: 'concat',
      version: 25
    });
  });

  it('verbose', () => {
    shouldHaveChanged(
      'concat',
      {
        name: 'sample #25',
        type: 'JSON',
        error: 'concat',
        version: 25
      },
      { verbose: true }
    );
  });
});

describe('multi rounds', () => {
  it('x2', () => {
    shouldHaveChanged('twoErrs', {
      name: 'sample #19',
      type: 'JSON',
      error: '2 errors',
      version: 19
    });
  });

  it('x2 (verbose)', () => {
    shouldHaveChanged(
      'twoErrs',
      {
        name: 'sample #19',
        type: 'JSON',
        error: '2 errors',
        version: 19
      },
      { verbose: true }
    );
  });

  it('x3', () => {
    shouldHaveChanged('threeErrs', {
      name: 'sample #21',
      type: 'JSON',
      error: '3 errors',
      version: 21
    });
  });
});

describe('special chars', () => {
  it('tab', () => {
    shouldHaveChanged('tab', {
      Test: '\t'
    });
  });

  it('formatted tab', () => {
    shouldHaveChanged('tabs', {
      Test: '\t'
    });
  });

  it('new line', () => {
    shouldHaveChanged('newLines', {
      Broken: '\n'
    });
  });
});
