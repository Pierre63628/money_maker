//+------------------------------------------------------------------+
//| DumpSymbolSpecs.mq5                                              |
//| Exports every tradable symbol's contract specification to a CSV.  |
//| Run it once on a DEMO account: Navigator -> Scripts -> double     |
//| click. The file lands in MQL5/Files/ (File -> Open Data Folder).  |
//| Then feed it to scripts/mt5_profile_from_specs.py.               |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

input string FilterSubstring = "";   // e.g. "BTC" to limit the dump; empty = every symbol
input bool   OnlyMarketWatch = false; // true = only the symbols shown in Market Watch
input string OutFile         = "symbol_specs.csv";

int OnStart()
  {
   int handle = FileOpen(OutFile, FILE_WRITE|FILE_CSV|FILE_ANSI, ';');
   if(handle == INVALID_HANDLE)
     {
      Print("cannot open ", OutFile, " error ", GetLastError());
      return(1);
     }
   FileWrite(handle, "symbol", "description", "path", "digits", "point", "spread_points",
             "spread_float", "contract_size", "volume_min", "volume_step", "volume_max",
             "swap_mode", "swap_long", "swap_short", "swap_rollover3days", "margin_initial",
             "trade_mode", "bid", "ask", "tick_value", "currency_base", "currency_profit");

   int total = OnlyMarketWatch ? SymbolsTotal(true) : SymbolsTotal(false);
   int written = 0;
   for(int i = 0; i < total; i++)
     {
      string name = SymbolName(i, OnlyMarketWatch);
      if(FilterSubstring != "" && StringFind(name, FilterSubstring) < 0)
         continue;
      if(!SymbolSelect(name, true))
         continue;
      long   trade_mode = SymbolInfoInteger(name, SYMBOL_TRADE_MODE);
      double bid        = SymbolInfoDouble(name, SYMBOL_BID);
      double ask        = SymbolInfoDouble(name, SYMBOL_ASK);
      FileWrite(handle, name,
                SymbolInfoString(name, SYMBOL_DESCRIPTION),
                SymbolInfoString(name, SYMBOL_PATH),
                (int)SymbolInfoInteger(name, SYMBOL_DIGITS),
                SymbolInfoDouble(name, SYMBOL_POINT),
                (int)SymbolInfoInteger(name, SYMBOL_SPREAD),
                (int)SymbolInfoInteger(name, SYMBOL_SPREAD_FLOAT),
                SymbolInfoDouble(name, SYMBOL_TRADE_CONTRACT_SIZE),
                SymbolInfoDouble(name, SYMBOL_VOLUME_MIN),
                SymbolInfoDouble(name, SYMBOL_VOLUME_STEP),
                SymbolInfoDouble(name, SYMBOL_VOLUME_MAX),
                (int)SymbolInfoInteger(name, SYMBOL_SWAP_MODE),
                SymbolInfoDouble(name, SYMBOL_SWAP_LONG),
                SymbolInfoDouble(name, SYMBOL_SWAP_SHORT),
                (int)SymbolInfoInteger(name, SYMBOL_SWAP_ROLLOVER3DAYS),
                SymbolInfoDouble(name, SYMBOL_MARGIN_INITIAL),
                (int)trade_mode, bid, ask,
                SymbolInfoDouble(name, SYMBOL_TRADE_TICK_VALUE),
                SymbolInfoString(name, SYMBOL_CURRENCY_BASE),
                SymbolInfoString(name, SYMBOL_CURRENCY_PROFIT));
      written++;
     }
   FileClose(handle);
   PrintFormat("wrote %d symbols to MQL5/Files/%s", written, OutFile);
   return(0);
  }
