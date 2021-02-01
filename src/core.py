import gpt_2_simple as gpt2
from datetime import datetime
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
session = gpt2.start_tf_sess()
gpt2.finetune(  session,
                dataset="enc.npz",
                model_name='117M',
                optimizer='sgd',
                batch_size=1,
                sample_length=256,
                learning_rate=0.0001,
                steps=200,
                restore_from='fresh',

                run_name='run1',
                print_every=10,
                sample_every=100,
                save_every=1000)