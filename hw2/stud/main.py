import utils
import os
import vocabulary



training_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/test_coarse_grained.json'))
#print(training_data)
#As vocabulary init class accept lists as input, this aux function does the conversion
list_candidates = utils.list_all_values(training_data,'candidates')
vocab = vocabulary.Vocabulary(training_data["words"],list_candidates)
