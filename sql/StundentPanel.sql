-- Where the stored procedures and functions will go
USE StudentPanel;

-- Stored procedure to get the tag for the inputed intent id
DELIMITER $$
DROP PROCEDURE IF EXISTS getTag$$

CREATE PROCEDURE getTag(
in myIntentId INT
)
BEGIN
	SELECT tag AS Tag
	FROM Intent
    WHERE intentId = myIntentId;
END $$
DELIMITER ;

-- Testing
-- CALL getTag(1);

-- Stored procedure to get the answer for the inputed intent id
DELIMITER $$
DROP PROCEDURE IF EXISTS getAnswer$$

CREATE PROCEDURE getAnswer(
in myIntentId INT
)
BEGIN
	SELECT answer AS Answer
	FROM Intent
    WHERE intentId = myIntentId;
END $$
DELIMITER ;

-- Testing
-- CALL getAnswer(1);

-- Stored procedure to get the medie type and url
DELIMITER $$
DROP PROCEDURE IF EXISTS getMedia$$

CREATE PROCEDURE getMedia(
in myIntentId INT
)
BEGIN
	SELECT media AS Type, url as URL
	FROM Intent
    WHERE intentId = myIntentId;
END $$
DELIMITER ;

-- Testing
-- CALL getMedia(35);

-- Stored procedure to get the patterns for the inputed question id
DELIMITER $$
DROP PROCEDURE IF EXISTS getKeywords$$

CREATE PROCEDURE getKeywords(
in myIntentId INT
)
BEGIN
	SELECT k.keyword AS Keywords
	FROM Intent AS i
    INNER JOIN Keywords AS k ON i.intentId = k.intentId
    WHERE i.intentId = myIntentId;
END $$
DELIMITER ;

-- Testing
-- CALL getKeywords(1);

-- Stored procedure to get number of questions
DELIMITER $$

DROP PROCEDURE IF EXISTS getNumberOfIntents $$

CREATE PROCEDURE getNumberOfIntents()
BEGIN
	SELECT COUNT(intentId) AS Number_Of_Intents
    FROM Intent;
END $$
DELIMITER ;

-- Tesing 
-- CALL getNumberOfIntents();
