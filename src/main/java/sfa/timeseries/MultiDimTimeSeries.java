package sfa.timeseries;

public class MultiDimTimeSeries {

    protected TimeSeries[] sources = null;
    protected int numSources = 0;
    protected String[] sourcesNames = null;
    protected Double labelMD = null;

    public MultiDimTimeSeries(TimeSeries[] data) {
        this.sources = data;
        this.numSources = data.length;
    }

    public MultiDimTimeSeries(TimeSeries[] data, Double label) {
        this(data);
        this.labelMD = label;
    }

    public MultiDimTimeSeries(TimeSeries[] data, String[] sourceNames, Double label) {
        this(data, label);
        this.sourcesNames = sourceNames;
        this.numSources = sourceNames.length;
    }

    /**
     * Get the label of MultiDim TimeSeries
     *
     * @return returns label of MultiDim TimeSeries
     */
    public Double getLabel() {
        return this.labelMD;
    }

    public void setLabel(Double label) {
        this.labelMD = label;
    }

    /**
     * Get the Lenght of main MultiDim TimeSeries
     *
     * @return returns the lenght of time series
     */
    public int getLength() {
        if (this.numSources != 0) {
            return this.sources[0].getLength();
        } else {
            return 0;

        }
    }

    /**
     * Get the total of sources
     *
     * @return returns the number of sources
     */
    public int getNumSources() {
        return this.sources.length;

    }

    /**
     * Get a unidimensional timeseries
     *
     * @param idSource the index of Dimension of MultiDim TimeSeries
     * @return returns instance od TimeSeries
     */
    public TimeSeries getTimeSeriesOfOneSource(int idSource){
        if(idSource >= numSources || idSource < 0){
            return null;
        }else{
            return new TimeSeries(sources[idSource].data,this.getLabel());
        }
    }
}
