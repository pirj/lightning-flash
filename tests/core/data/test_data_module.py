# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from pytorch_lightning.trainer.states import RunningStage

from flash import FlashDataset, PreTransform, Task, Trainer
from flash.core.data.new_data_module import DataModule


def test_new_datamodule():
    class TestDataset(FlashDataset):

        pass

    train_dataset = TestDataset(RunningStage.TRAINING)
    train_dataset.pass_args_to_load_data(range(10))

    dm = DataModule(train_dataset=train_dataset)
    trainer = Trainer(fast_dev_run=True)
    model = Task(torch.nn.Linear(1, 2), loss_fn=torch.sum())
    trainer.fit(model, dm)
