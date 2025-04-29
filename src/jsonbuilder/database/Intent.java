package jsonbuilder.database;

import java.util.ArrayList;

public class Intent {
    private String tag;
    private String answer;
    private Media media;
    private ArrayList<String> keywords;

    public Intent(String question, String answer, ArrayList<String> patterns, Media media) {
        this.tag = question;
        this.answer = answer;
        this.keywords = patterns;
        this.media = media;
    }

    public String getTag() {
        return this.tag;
    }

    public String getAnswer() {
        return this.answer;
    }

    public ArrayList<String> getKeywords() {
        return this.keywords;
    }

    public Media getMedia(){
        return this.media;
    }

}
