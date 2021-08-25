from RobFR.dataset.cfp import CFPLoader
from RobFR.dataset.ytf import YTFLoader
from RobFR.dataset.lfw import LFWLoader
LOADER_DICT = {
    'lfw': LFWLoader,
    'ytf': YTFLoader,
    'cfp': CFPLoader
}