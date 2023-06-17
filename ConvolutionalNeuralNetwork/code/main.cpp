#include <iostream>
#include "../leonardo_overlord.hpp"
#include "../chess_arena.hpp"
#include "../../MockChessEngine/RandomPlayer.h"
#include "../../MockChessEngine/AlphaBetaPruningBot.h"
#include "../leonardo_bot.hpp"
#include "../../MockChessEngine/Game.h"
int main()
{
	leonardo_overlord overlord("BetaZero");
	overlord.train();

	return 0;
}