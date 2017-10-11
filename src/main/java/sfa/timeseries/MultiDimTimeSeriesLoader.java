package sfa.timeseries;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 */
public class MultiDimTimeSeriesLoader {

    public static MultiDimTimeSeries[] loadDataset(File[] datasets, File labels) throws IOException {
        int numOfInstancias = 0;
        int numSources = datasets.length;
        ArrayList<Double> labelsForEachWindowResult = new ArrayList<Double>();
        ArrayList<MultiDimTimeSeries> instancias = new ArrayList<MultiDimTimeSeries>();

        try (BufferedReader br = new BufferedReader(new FileReader(labels))) {
            String line = null;
            String[] labelsForEachWindow = null;

            while ((line = br.readLine()) != null) {
                labelsForEachWindow = line.split(",");
                int j = 0;

                for (int i = 0; i < labelsForEachWindow.length; i++) {
                    String column = labelsForEachWindow[i].trim();
                    try {
                        if (isNonEmptyColumn(column)) {
                            Double columnDouble = Double.parseDouble(column);
                            labelsForEachWindowResult.add(columnDouble);
                        }
                    } catch (NumberFormatException nfe) {
                        nfe.printStackTrace();
                    }
                }
                numOfInstancias = labelsForEachWindowResult.size();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Signals
        BufferedReader[] readers = new BufferedReader[numSources];

        for (int sourceId = 0; sourceId < numSources; sourceId++) {

            readers[sourceId] = new BufferedReader(new FileReader(datasets[sourceId]));

        }

        try {
            for (int iLine = 0; iLine < numOfInstancias; iLine++) {
                double[][] lines = new double[numSources][];
                for (int ireaders = 0; ireaders < numSources; ireaders++) {
                    String line = readers[ireaders].readLine();
                    if (line != null) {
                        String[] columns = line.split(",");
                        double[] dataOfOneLine = new double[columns.length];
                        int j = 0;
                        for (int i = 0; i < columns.length; i++) {
                            String column = columns[i].trim();
                            try {
                                if (isNonEmptyColumn(column)) {
                                    dataOfOneLine[j++] = Double.parseDouble(column);
                                }
                            } catch (NumberFormatException nfe) {
                                nfe.printStackTrace();
                            }
                        }
                        lines[ireaders] = dataOfOneLine;
                    }
                }
                TimeSeries[] tsdata = new TimeSeries[numSources];
                for (int i = 0; i < numSources; i++) {
                    tsdata[i] = new TimeSeries(Arrays.copyOfRange(lines[i], 0, lines[i].length));
                }

                MultiDimTimeSeries tsMD = new MultiDimTimeSeries(tsdata, labelsForEachWindowResult.get(iLine));
                instancias.add(tsMD);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        //System.out.println("Done reading from " + dataset + " samples " + samples.size() + " length " + samples.get(0).getLength());
        return instancias.toArray(new MultiDimTimeSeries[]{});
    }

    public static boolean isNonEmptyColumn(String column) {
        return column != null && !"".equals(column) && !"NaN".equals(column) && !"\t".equals(column);
    }

    /*public static MultiDimTimeSeries generateRandomWalkData(int maxDimension, Random generator) {
        double[] data = new double[maxDimension];

        // Gaussian Distribution
        data[0] = generator.nextGaussian();

        for (int d = 1; d < maxDimension; d++) {
            data[d] = data[d - 1] + generator.nextGaussian();
        }

        return new TimeSeries(data);
    }*/
}
