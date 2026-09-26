"""Definizione formale degli intenti semantici supportati da Makima."""

from enum import Enum


class Intent(str, Enum):
    """Intento categorico della richiesta dell'utente.
    
    Inclusione esplicita di UNKNOWN per gestire richieste non interpretabili o ambigue
    senza allucinazioni.
    """
    QUERY = "QUERY"                     # Richiesta di previsione o stima probabilistica
    COMMAND = "COMMAND"                 # Istruzione o comando operativo (es. "aggiorna", "esegui")
    INFORMATION = "INFORMATION"         # Richiesta di spiegazione o dettagli descrittivi
    TEMPORAL_QUERY = "TEMPORAL_QUERY"   # Interrogazione puramente focalizzata su orizzonti/scadenze
    OBSERVATION = "OBSERVATION"         # Registrazione di un fatto empirico o esito
    STATUS = "STATUS"                   # Richiesta di stato diagnostico del motore
    UNKNOWN = "UNKNOWN"                 # Intento non riconosciuto, ambiguo o fuori dominio
