import java.sql.SQLException;
import java.util.ArrayList;

public class BusinessLayer {
    DataMgr dataMgr;

    public BusinessLayer(String username, String password) {
        this.dataMgr = new DataMgr(username, password);
        try {
            dataMgr.getConnection();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public String getQuestion(int myQuestionId) {
        return this.dataMgr.getQuestion(myQuestionId);
    }

    public String getAnswer(int myQuestionId) {
        return this.dataMgr.getAnswer(myQuestionId);
    }

    public ArrayList<String> getPatterns(int myQuestionId){
        return this.dataMgr.getPatterns(myQuestionId);
    }

    public int getNumberOfQuestions() {
        return this.dataMgr.getNumberOfQuestions();
    }

}
