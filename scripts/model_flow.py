from metaflow import FlowSpec, step, Parameter

class ModelFlow(FlowSpec):

    epochs = Parameter('epochs', default=10)

    @step
    def start(self):
        print("Data loading and preprocessing...")
        # Load and preprocess data
        self.data = ...
        self.next(self.train_model)

    @step
    def train_model(self):
        print("Training model...")
        # Train model
        self.model = ...
        self.next(self.predict)

    @step
    def predict(self):
        print("Running predictions...")
        # Generate predictions
        self.predictions = self.model.predict(self.data)
        self.next(self.end)

    @step
    def end(self):
        print("Flow complete.")