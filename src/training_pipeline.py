import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import yaml
from src.tft_model import TFTModel
from src.dataset_builder import DatasetBuilder
from src.feature_engineer import FeatureEngineer
from src.forecast_repository import ForecastRepository

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_training_pipeline(config_path: str):
    config = load_config(config_path)

    # Initialize components
    feature_engineer = FeatureEngineer(config)
    repository = ForecastRepository(config)
    dataset_builder = DatasetBuilder(config, feature_engineer)

    # Load and process data
    raw_df = dataset_builder.load_raw_data(config['data']['raw_path'])
    processed_df = dataset_builder.preprocess_data(raw_df)
    featured_df = feature_engineer.engineer_features(processed_df)
    featured_df = featured_df.dropna().reset_index(drop=True)

    # Build datasets
    train_dataset, val_dataset, test_dataset = dataset_builder.build_datasets(featured_df)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['model']['batch_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['model']['batch_size'])

    # Initialize model
    model = TFTModel(config, train_dataset)

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=config['training']['early_stopping_patience'])
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['data']['models_path'],
        filename='tft-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['model']['max_epochs'],
        callbacks=[early_stopping, checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    # Train
    trainer.fit(model, train_dataloader, val_dataloader)

    # Test
    trainer.test(model, test_dataloader)

    # Save final model
    repository.save_model_checkpoint(model, trainer.optimizers[0], trainer.current_epoch, "final_model")

    print("Training completed.")

if __name__ == "__main__":
    config_path = "config/config.yaml"
    run_training_pipeline(config_path)