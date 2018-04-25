class Transformer:
    is_model = False

    def __init__(self, name):
        self.name = name
        self.transformation = None

    def __repr__(self):
        return self.name

    def transform(self, data):
        raise NotImplementedError


class CompositeTransformer(Transformer):
    """A composite for transformers"""
    def __init__(self, name):
        super().__init__(name)
        self.transformers = []

    def add(self, transformer, i=None):
        if i:
            self.transformers.insert(i, transformer)
        else:
            self.transformers.append(transformer)

    def remove(self, transformer_name):
        for transformer in self.transformers:
            if transformer.name == transformer_name:
                self.transformers.remove(transformer)
                break

    def transform(self, input):
        pass


class SequentialTransformer(CompositeTransformer):
    def transform(self, data):
        output = data
        for transformer in self.transformers:
            output = transformer.transform(output)
        return output


class ParallelTransformer(CompositeTransformer):
    def transform(self, data):
        output = []
        for transformer in self.transformers:
            output.append(transformer.transform(data))
        return output


class Pipeline(SequentialTransformer):

    def fit(self, data, k=1):
        output = data
        for transformer in self.transformers:
            if transformer._transformer_type == 'model':
                transformer.fit(output, k=k)
            else:
                output = transformer.transform(output)
        return self

    def predict(self, data):
        output = data
        for transformer in self.transformers:
            if transformer._transformer_type == 'model':
                output = transformer.predict(output)
            else:
                output = transformer.transform(output)
        return output

    def validate(self, data, filtered_items=None):
        output = data
        for transformer in self.transformers:
            if transformer._transformer_type == 'model':
                output = transformer.validate(output, filtered_items)
            else:
                output = transformer.transform(output)
        return output
