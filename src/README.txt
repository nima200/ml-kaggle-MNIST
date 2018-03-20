Project Members:
-   Name: Nima Adibpour
    Student ID: 260606511
    Email: nima.adibpour@mail.mcgill.ca

-   Name: Charlotte Ding
    Student ID: 260606835
    Email: xiaoye.ding@mail.mcgill.ca

-   Name: Pavel Kondratyev
    Student ID: 260653115
    Email: pavel.kondratyev@mail.mcgill.ca



    Our SVM Implementation is under the "Naive SVM.ipynb" file (for part 1)
    Our Neural Network classifier implementation is under NNClassifier.py, driver code that uses it to fit and predict
        is under "kaggle_nn.ipynb" (for part 2)
    Our CNN Implementation is done using Keras and is under "kaggle_cnn.ipynb" (for part 3)
    We have created extra models than expected in the project, these extra models are KNN under Naive "K-NN.ipynb",
        KNN with PCA under "PCA + K-NN.ipynb" file, Random Forest implementation under "RandomForest.ipynb" file, and
        SVM with non linear kernels under "SVM-Different Kernels.ipynb" file (for this one we haven't reported results on
        as it takes extremely long to calculate on CPU.

    Pre Processing is implemented as methods in "pre_processing.py", called by all notebooks when needed.
    Data Augmentation (although not used in notebooks as was discarded for final results, yet reported on) is under the
        "data_augmentation.py" file which returns a Keras generator.