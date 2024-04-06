import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';
import { mdiPlayCircle, mdiReload } from '@mdi/js';

const numRows = 6;
const numCols = 7;

const App = () => {
  const [board, setBoard] = useState(Array(numRows).fill().map(() => Array(numCols).fill(0)));
  const [currentPlayer, setCurrentPlayer] = useState('yellow');
  const [winner, setWinner] = useState(null);
  const [winningSequence, setWinningSequence] = useState([]);
  const [gameStarted, setGameStarted] = useState(false);
  const [randomMoves, setRandomMoves] = useState(false);
  const [mctsIterations, setMctsIterations] = useState(50);

  const checkWinner = useCallback((row, col, player) => {
    const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
    for (let [dx, dy] of directions) {
      let count = 1;
      const sequence = [[row, col]];
      for (let dir = -1; dir <= 1; dir += 2) {
        for (let i = 1; i < 4; i++) {
          const r = row + dir * i * dx;
          const c = col + dir * i * dy;
          if (r < 0 || r >= numRows || c < 0 || c >= numCols || board[r][c] !== player) break;
          count++;
          sequence.push([r, c]);
        }
      }
      if (count >= 4 && !winner) {
        console.log("Winner is ", "Winning Sequence:", sequence); // Debugging
        setWinningSequence(sequence);
        return true;
      }
    }
    return false;
  }, [setWinningSequence, board, winner]);


  const make_move = useCallback((col) => {
    const newBoard = [...board];
    for (let i = numRows - 1; i >= 0; i--) {
      if (!newBoard[i][col]) {
        newBoard[i][col] = currentPlayer === 'red' ? 1 : -1;
        if (checkWinner(i, col, currentPlayer === 'red' ? 1 : -1)) {
          setWinner(currentPlayer);
        } else {
          setCurrentPlayer(currentPlayer === 'red' ? 'yellow' : 'red');
          // console.log("Current player set to ", currentPlayer)
        }
        setBoard(newBoard);
        return;
      }
    }
  }, [board, currentPlayer, checkWinner]);

  useEffect(() => {
    if (currentPlayer === 'yellow' && !winner && gameStarted) {
      // Make API call
      axios.post('http://127.0.0.1:8000/get_move',
        { 'board': board, currentPlayer: currentPlayer, randomMoves: randomMoves, mctsIterations: mctsIterations})
        .then(response => {
          console.log(response.data); // Handle response data as needed
          make_move(response.data.move)
        })
        .catch(error => {
          console.error('Error fetching data:', error);
        });
    }
  }, [currentPlayer, board, make_move, winner, randomMoves, mctsIterations, gameStarted]);


  const handleClick = (col) => {
    if (!gameStarted) {
      setGameStarted(true);
    }
    else if (winner || !gameStarted) return; // Game already won
    make_move(col)
  };

  const handleReload = () => {
    setBoard(Array(numRows).fill().map(() => Array(numCols).fill(0)));
    setCurrentPlayer('yellow');
    setWinner(null);
    setWinningSequence([]);
    setGameStarted(false);
    setRandomMoves(false);
    setMctsIterations(50);
  };

  const handleRandomMovesChange = (event) => {
    setRandomMoves(event.target.checked);
  };

  const handleMctsIterationsChange = (event) => {
    setMctsIterations(event.target.value);
  };

  const renderCell = (row, col) => {
    const discColor = board[row][col] !== 0 ? (board[row][col] === 1 ? 'red' : 'yellow') : null;
    const backgroundColor = discColor ? discColor : 'white';
    const isWinningCell = winningSequence.some(([r, c]) => r === row && c === col);

    return (
      <div
        className={`cell ${isWinningCell ? 'winning-cell' : ''}`}
        key={`${row}-${col}`}
        onClick={() => handleClick(col)}
        style={{ backgroundColor }}
      />
    );
  };

  const renderBoard = () => (
    <div className="board">
      {board.map((row, rowIndex) => (
        <div key={rowIndex} className="row">
          {row.map((_, colIndex) => renderCell(rowIndex, colIndex))}
        </div>
      ))}
    </div>
  );

  return (
    <div className="App">
      <h1>Connect 4</h1>
      <div className="options-container">
        <div>
          <input
            type="checkbox"
            id="randomMoves"
            name="randomMoves"
            checked={randomMoves}
            onChange={handleRandomMovesChange}
          />
          <label htmlFor="randomMoves">Get Random Moves</label>
        </div>
        <div>
          <input
            type="range"
            id="mctsIterations"
            name="mctsIterations"
            min="50"
            max="500"
            value={mctsIterations}
            onChange={handleMctsIterationsChange}
          />
          <label htmlFor="mctsIterations">MCTS Iterations: {mctsIterations}</label>
        </div>
      </div>
      {renderBoard()}
      {gameStarted && winner && (
        <div className="reload-button" onClick={handleReload}>
          <svg className="reload-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d={mdiReload} />
          </svg>
        </div>
      )}
      {!gameStarted && !winner && (
        <div className="play-button-container">
          <div className="play-button" onClick={() => setGameStarted(true)}>
            <svg className="play-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
              <path d={mdiPlayCircle} />
            </svg>
          </div>
        </div>
      )}
    </div>
  );
  
};

export default App;
