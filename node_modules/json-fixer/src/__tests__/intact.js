const { exam } = require('../test.utils');

describe('keeps a correct file intact', () => {
  it('normal file', () => {
    exam({
      sampleName: 'normal',
      expectedOutput: {
        name: 'sample #0',
        type: 'JSON',
        version: 0
      }
    });
  });

  it('floating points', () => {
    exam({
      sampleName: 'fp',
      expectedOutput: {
        name: 'sample #2',
        type: 'JSON',
        version: 2.0
      }
    });
  });
});

test('boolean option', () => {
  exam({
    sampleName: 'normal',
    expectedOutput: {
      name: 'sample #0',
      type: 'JSON',
      version: 0
    },
    fixerOptions: true
  });
});
