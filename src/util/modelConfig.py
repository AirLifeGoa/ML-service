import yaml
import os
from os import path

print(os.path.dirname(os.path.realpath(__file__)))

class ModelConfigClient:

    def __init__(self):
        pass

    def check_config_file(self,id):
        if not path.exists(f"./configs/station_{str(id)}.yaml"):
            return False
        return True
    
    def check_model_config(self,id,model):
        with open(f'./configs/station_{str(id)}.yaml','r') as f:
            output = yaml.safe_load(f)
        print(output)
        if output == None or not model in output.keys():
            return False
        return True

    def create_config_file(self,id,model):
        with open(f'./configs/station_{str(id)}.yaml', 'w',) as f:
            pass
        print(f'Created a config file for station {id}')

    def load_default_config(self,model):
        with open(f'./configs/default_config.yaml','r') as f:
            output = yaml.safe_load(f)
        print(output)
        return output[model]

    def write_default_config(self,id,model):
        with open(f'./configs/station_{str(id)}.yaml','a') as f:
            output = yaml.dump({model: self.load_default_config(model)},f,sort_keys=False)
        
    
    def load_model_config(self,id,model):
        try:
                
            if not self.check_config_file(id):
                self.create_config_file(id,model)
            
            if not self.check_model_config(id,model):

                self.write_default_config(id,model)

            with open(f'./configs/station_{str(id)}.yaml','r') as f:
                output = yaml.safe_load(f)

            return output[model]
        except:
            print("Unable to load config file")
            return None
    

if __name__ == "__main__":
    obj = ModelConfigClient()
    obj.load_model_config("25072002","prophet_model") 
    obj.load_model_config("25072002","prophet2_model") 