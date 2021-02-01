import gpt_2_simple as gpt2
from datetime import datetime

gpt2.finetune(  gpt2.start_tf_sess,
                dataset="enc.npz",
                model_name='124M',
                steps=1000,
                restore_from='fresh',
                run_name='run1',
                print_every=100,
                sample_every=100,
                save_every=1000)
