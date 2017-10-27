package sfa;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.classification.*;
import sfa.timeseries.MultiDimTimeSeries;
import sfa.timeseries.MultiDimTimeSeriesLoader;

import java.io.File;
import java.io.IOException;

@RunWith(JUnit4.class)
public class MDClassificationTest {

    // The datasets to use
    public static String[] datasets = new String[]{
            "UCI/UCI-HAR-TOTAL-REAL",
    };

    @Test
    public void testMDClassification() throws IOException {
        // the relative path to the datasets
        ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

        File dir = new File(classLoader.getResource("datasets/").getFile());


        for (String s : datasets) {
            File d = new File(dir.getAbsolutePath() + "/" + s);
            if (d.exists() && d.isDirectory()) {

                boolean withGYRO = false;
                int num_sources = 0;
                File x_train = new File(d.getAbsolutePath() + "/" + "X_TRAIN");
                File y_train = new File(d.getAbsolutePath() + "/" + "Y_TRAIN");
                File z_train = new File(d.getAbsolutePath() + "/" + "Z_TRAIN");

                File x_test = new File(d.getAbsolutePath() + "/" + "X_TEST");
                File y_test = new File(d.getAbsolutePath() + "/" + "Y_TEST");
                File z_test = new File(d.getAbsolutePath() + "/" + "Z_TEST");

                File labels_train = new File(d.getAbsolutePath() + "/" + "LABELS_TRAIN");
                File labels_test = new File(d.getAbsolutePath() + "/" + "LABELS_TEST");

                File gx_train = null;
                File gy_train = null;
                File gz_train = null;

                File gx_test = null;
                File gy_test = null;
                File gz_test = null;

                if (withGYRO) {
                    num_sources = 6;
                    gx_train = new File(d.getAbsolutePath() + "/GYRO/" + "X_TRAIN");
                    gy_train = new File(d.getAbsolutePath() + "/GYRO/" + "Y_TRAIN");
                    gz_train = new File(d.getAbsolutePath() + "/GYRO/" + "Z_TRAIN");

                    gx_test = new File(d.getAbsolutePath() + "/GYRO/" + "X_TEST");
                    gy_test = new File(d.getAbsolutePath() + "/GYRO/" + "Y_TEST");
                    gz_test = new File(d.getAbsolutePath() + "/GYRO/" + "Z_TEST");
                } else {
                    num_sources = 3;
                }
                File[] sources_train = new File[num_sources];
                File[] sources_test = new File[num_sources];


                sources_train[0] = x_train;
                sources_train[1] = y_train;
                sources_train[2] = z_train;

                sources_test[0] = x_test;
                sources_test[1] = y_test;
                sources_test[2] = z_test;

                if (withGYRO) {
                    sources_train[3] = gx_train;
                    sources_train[4] = gy_train;
                    sources_train[5] = gz_train;

                    sources_test[3] = gx_test;
                    sources_test[4] = gy_test;
                    sources_test[5] = gz_test;
                }
                Classifier.DEBUG = false;

                // Load the train/test splits
                MultiDimTimeSeries[] trainSamples = MultiDimTimeSeriesLoader.loadDataset(sources_train, labels_train);
                MultiDimTimeSeries[] testSamples = MultiDimTimeSeriesLoader.loadDataset(sources_test, labels_test);

                // The BOSS VS classifier
                MDClassifier bossMD = new BOSSMDVSClassifier();
                MDClassifier.Score scoreBOSSVS = bossMD.eval(trainSamples, testSamples);
                System.out.println(s + ";" + scoreBOSSVS.toString());


            } else {
                // not really an error. just a hint:
                System.out.println("Dataset could not be found: " + d.getAbsolutePath() + ". " +
                        "Please download datasets from [http://www.cs.ucr.edu/~eamonn/time_series_data/].");
            }
        }

    }
}
