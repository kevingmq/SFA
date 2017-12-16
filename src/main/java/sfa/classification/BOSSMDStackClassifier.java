package sfa.classification;

import com.carrotsearch.hppc.IntFloatHashMap;
import com.carrotsearch.hppc.ObjectObjectHashMap;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;
import sfa.transformation.BOSSMDStackVS;
import sfa.transformation.BOSSMDStackVS.BagOfPattern;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

public class BOSSMDStackClassifier extends Classifier{

    // default training parameters
    public static double factor = 0.90;

    public static int maxF = 10;
    public static int minF = 10;
    public static int maxS = 4;



    public BOSSMDStackClassifier(){
        super();
    }

    public static class BOSSMDStackModel<E> extends Model {

        public BOSSMDStackModel(
                boolean normed,
                int windowLength) {
            super("BOSSMDStackModel", -1, 1, -1, 1, normed, windowLength);
        }

        // The inverse document frequencies learned by training
        public ObjectObjectHashMap<Double, E> idf;

        // the trained BOSS MD transformation
        public BOSSMDStackVS bossmd;

        // the best number of Fourier values to be used
        public int features;
    }


    @Override
    public Score eval(
            final TimeSeries[] trainSamples, final TimeSeries[] testSamples) {
        throw new RuntimeException("Please use: eval(" +
                "final MultiVariateTimeSeries[] trainSamples, final MultiVariateTimeSeries[] testSamples)");
    }

    @Override
    public Double[] predict(TimeSeries[] samples) {
        throw new RuntimeException("Please use: predict(final MultiVariateTimeSeries[] samples)");
    }

    @Override
    public Score fit(final TimeSeries[] trainSamples) {
        throw new RuntimeException("Please use: fit(final MultiVariateTimeSeries[] trainSamples)");
    }

    @Override
    public Predictions score(final TimeSeries[] testSamples) {
        throw new RuntimeException("Please use: score(final MultiVariateTimeSeries[] testSamples)");
    }


    public Score fit(final MultiVariateTimeSeries[] trainSamples) {
        // generate test train/split for cross-validation
        generateIndices(trainSamples);

        Score bestScore = null;
        int bestCorrectTraining = 0;

        int minWindowLength = 10;
        int maxWindowLength = getMax(trainSamples,MAX_WINDOW_LENGTH);

        // equi-distance sampling of windows
        ArrayList<Integer> windows = new ArrayList<>();
        double count = Math.sqrt(maxWindowLength);
        double distance = ((maxWindowLength - minWindowLength) / count);
        //distance = 1;
        for (int c = minWindowLength; c <= maxWindowLength; c += distance) {
            windows.add(c);
        }

        for (boolean normMean : NORMALIZATION) {
            // train the shotgun models for different window lengths
            Ensemble<BOSSMDStackModel<IntFloatHashMap>> model = fitEnsemble(windows.toArray(new Integer[]{}), normMean, trainSamples);

            //Double[] labels = predict(model, trainSamples);
            //Predictions pred = evalLabels(trainSamples, labels);

            /*if (bestCorrectTraining <= pred.correct.get()) {
                bestCorrectTraining = pred.correct.get();
                bestScore = model.getHighestScoringModel().score;
                bestScore.training = pred.correct.get();
                this.model = model;
            }*/
        }

        return null;
    }

    public Predictions score(final MultiVariateTimeSeries[] testSamples) {
        return null;
    }


    public Double[] predict(final MultiVariateTimeSeries[] testSamples) {
        return new Double[0];
    }

    public Score eval(final MultiVariateTimeSeries[] trainSamples, final MultiVariateTimeSeries[] testSamples) {
        return null;
    }
    public Score evalCrossValidation(final MultiVariateTimeSeries[] trainSamples) {
        long startTime = System.currentTimeMillis();

        Score score = fit(trainSamples);

        return null;
    }

    protected Ensemble<BOSSMDStackModel<IntFloatHashMap>> fitEnsemble(Integer[] windows,
                                                                 boolean normMean,
                                                                 MultiVariateTimeSeries[] samples) {

        final List<BOSSMDStackModel<IntFloatHashMap>> results = new ArrayList<>(windows.length);
        final AtomicInteger correctTraining = new AtomicInteger(0);

        ParallelFor.withIndex(exec, threads, new ParallelFor.Each() {
            Set<Double> uniqueLabels = uniqueClassLabels(samples);

            @Override
            public void run(int id, AtomicInteger processed) {
                for (int i = 0; i < windows.length; i++) {
                    if (i % threads == id) {
                        BOSSMDStackModel<IntFloatHashMap> model = new BOSSMDStackModel<>(normMean, windows[i]);
                        try {
                            BOSSMDStackVS bossmd = new BOSSMDStackVS(maxF, maxS, windows[i], model.normed);
                            int[/*samples */][/*source */][/*wordInt*/] words = bossmd.createWords(samples);


                            for (int f = minF; f <= Math.min(model.windowLength, maxF); f += 2) {
                                BagOfPattern[] bag = bossmd.createBagOfPattern(words, samples, f);

                               // cross validation using folds
                                int correct = 0;
                              /*  for (int s = 0; s < folds; s++) {
                                    // calculate the tf-idf for each class
                                    ObjectObjectHashMap<Double, IntFloatHashMap> idf = bossmd.createTfIdf(bag,
                                            BOSSMDStackClassifier.this.trainIndices[s], this.uniqueLabels);

                                    correct += predict(BOSSMDStackClassifier.this.testIndices[s], bag, idf).correct.get();
                                }
                                if (correct > model.score.training) {
                                    model.score.training = correct;
                                    model.features = f;

                                    if (correct == samples.length) {
                                        break;
                                    }
                                }*/
                            }

                            // obtain the final matrix
                           /* BagOfPattern[] bag = bossmd.createBagOfPattern(words, samples, model.features);

                            // calculate the tf-idf for each class
                            model.idf = bossmd.createTfIdf(bag, this.uniqueLabels);
                            model.bossmd = bossmd;*/

                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                        // keep best scores
                       /* synchronized (correctTraining) {
                            if (model.score.training > correctTraining.get()) {
                                correctTraining.set(model.score.training);
                            }

                            // add to ensemble if train-score is within factor to the best score
                            if (model.score.training >= correctTraining.get() * factor) {
                                results.add(model);
                            }
                        }*/
                    }
                }
            }
        });

        // returns the ensemble based on the best window-lengths within factor
        return filterByFactor(results, correctTraining.get(), factor);
    }
}
