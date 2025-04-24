-- Where the stored procedures and functions will go
USE StudentPanel;

-- Stored procedure to get the question for the inputed question id
DELIMITER $$
DROP PROCEDURE IF EXISTS getQuestion$$

CREATE PROCEDURE getQuestion(
in myQuestionId INT
)
BEGIN
	SELECT q.question AS Question
	FROM Questions AS q
    WHERE q.questionId = myQuestionId;
END $$
DELIMITER ;

-- Testing
-- CALL getQuestion(1);

-- Stored procedure to get the answer for the inputed question id
DELIMITER $$
DROP PROCEDURE IF EXISTS getAnswer$$

CREATE PROCEDURE getAnswer(
in myQuestionId INT
)
BEGIN
	SELECT q.answer AS Answer
	FROM Questions AS q
    WHERE q.questionId = myQuestionId;
END $$
DELIMITER ;

-- Testing
-- CALL getAnswer(1);

-- Stored procedure to get the patterns for the inputed question id
DELIMITER $$
DROP PROCEDURE IF EXISTS getPatterns$$

CREATE PROCEDURE getPatterns(
in myQuestionId INT
)
BEGIN
	SELECT pa.pattern AS Patterns
	FROM Questions AS q
    INNER JOIN Patterns AS pa ON q.questionId = pa.questionId
    WHERE q.questionId = myQuestionId;
END $$
DELIMITER ;

-- Testing
-- CALL getPatterns(1);

-- Stored procedure to get number of questions
DELIMITER $$

DROP PROCEDURE IF EXISTS getNumberOfQuestions $$

CREATE PROCEDURE getNumberOfQuestions()
BEGIN
	SELECT COUNT(questionId) AS Number_Of_Questions
    FROM Questions;
END $$
DELIMITER ;

-- Tesing 
CALL getNumberOfQuestions();
