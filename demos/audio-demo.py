import os

import kagglehub

import palimpzest as pz


class SmallAudioDataset(pz.AudioFileDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Limit to first 10 audio files for demo purposes
        self.filepaths = self.filepaths[:10]


if __name__ == "__main__":
    # Download latest version
    path = kagglehub.dataset_download("rushibalajiputthewad/sound-classification-of-animal-voice")
    print(f"Dataset downloaded to: {path}")

    # create simple plan to classify animal sounds
    plan = SmallAudioDataset(id="animal-sounds", path=os.path.join(path, "Animal-Soundprepros"))
    plan = plan.sem_map(cols=[{"name": "animal", "type": str, "description": "The type of animal making the sound in the recording."}])

    # run plan un-optimized
    config = pz.QueryProcessorConfig(policy=pz.MaxQuality())
    output = plan.run(config)

    print(output.to_df())
