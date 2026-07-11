from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from src.models.custom_layers import AbsoluteDifference


def build_siamese_model(embedding_dim=384):
    resume_input = Input(shape=(embedding_dim,), name="resume_embedding")
    jd_input = Input(shape=(embedding_dim,), name="jd_embedding")

    shared_dense_1 = Dense(256, activation="relu")
    shared_dropout = Dropout(0.3)
    shared_dense_2 = Dense(128, activation="relu")

    resume_vector = shared_dense_1(resume_input)
    resume_vector = shared_dropout(resume_vector)
    resume_vector = shared_dense_2(resume_vector)

    jd_vector = shared_dense_1(jd_input)
    jd_vector = shared_dropout(jd_vector)
    jd_vector = shared_dense_2(jd_vector)

    difference = AbsoluteDifference()([resume_vector, jd_vector])

    combined = Concatenate()([resume_vector, jd_vector, difference])

    x = Dense(128, activation="relu")(combined)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)

    output = Dense(1, activation="sigmoid", name="match_score")(x)

    model = Model(inputs=[resume_input, jd_input], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model