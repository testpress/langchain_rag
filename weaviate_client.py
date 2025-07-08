import weaviate, os, getpass
from weaviate.classes.init import Auth

class WeaviateDB:
    def __init__(self):
        self._client = None
        self._configure_environment()
    
    def _configure_environment(self):
        self.http_host = os.environ.get("WEAVIATE_HTTP_HOST", "localhost")
        self.grpc_host = os.environ.get("WEAVIATE_GRPC_HOST", "localhost")
        self.http_port = int(os.environ.get("WEAVIATE_HTTP_PORT", "8080"))
        self.grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
        self.weaviate_use_ssl = os.environ.get("WEAVIATE_USE_SSL", "false")
        
        if not os.environ.get("WEAVIATE_API_KEY"):
            os.environ["WEAVIATE_API_KEY"] = getpass.getpass("Enter API key for Weaviate: ")
        self.weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
    
    def get_client(self):
        if self._client is None:
            self._client = weaviate.connect_to_custom(
                http_host=self.http_host,        
                http_port=self.http_port,              
                http_secure=self.weaviate_use_ssl,           
                grpc_host=self.grpc_host,        
                grpc_port=self.grpc_port,              
                grpc_secure=self.weaviate_use_ssl,           
                auth_credentials=Auth.api_key(self.weaviate_api_key),    
            )
        return self._client


if __name__ == "__main__":
    client_manager = WeaviateDB()
    weaviate_client = client_manager.get_client()
    print("Weaviate client initialized successfully!")
    
    
    
