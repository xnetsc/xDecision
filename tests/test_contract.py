import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xdecision.prepare import prepare
from xdecision.train import collate
from xdecision.runtime import Model


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'data.jsonl'

    def tearDown(self):
        self.tmp.cleanup()

    def encode(self, target, **extra):
        self.path.write_text(json.dumps(dict(state='facts',question=dict(type='choice',instructions='pick',criteria={'a':'A','b':'B'}),target=target,**extra))+'\n')
        with patch('xdecision.prepare.load_config',return_value={'max_len':1024,'head_max_len':256}), patch('xdecision.prepare.load_tokenizer'), patch('xdecision.prepare.build_sequence',return_value=([2,4,5,4,6,1],[1,3])):
            return prepare(self.path,'unused')

    def test_valid_distribution_and_explicit_uncertainty(self):
        rows=self.encode([0.5,0.5],determinate=False)
        self.assertFalse(rows[0]['determinate'])
        self.assertEqual(collate(rows,0)['target'][0].tolist(),[0.5,0.5])

    def test_invalid_distribution(self):
        for target in ([0.5,0.6],[-0.1,1.1],[float('nan'),0],[1]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.encode(target)

    def test_invalid_label(self):
        with self.assertRaises(ValueError):
            self.encode([1,0],label=2)

    def test_runtime_changes_name_only(self):
        from unittest.mock import Mock
        model = Model.__new__(Model)
        answers = {'q': {'probabilities': {'a': 0.6, 'b': 0.4}, 'confidence': 0.02}}
        model._model = Mock()
        model._model.predict.return_value = {'model': 'upstream', 'answers': answers}
        result = model.predict('facts', {})
        self.assertEqual(result['model'], 'xDecision')
        self.assertIs(result['answers'], answers)


if __name__=='__main__':
    unittest.main()
