import torch
from train_affinity_predictor import CombinedModel
import numpy as np
from transformers import AutoTokenizer, AutoModel

from scripts_with_other_functions.get_hyperparams_db import get_best_trial

def predict_affinity(smile, fasta, path_model, best_trial):
    model = CombinedModel(hidden_dim=best_trial["params"]["hidden_dim"],
                          num_layers=best_trial["params"]["num_layers"],
                          dropout=best_trial["params"]["dropout"],
                          n_attention_heads=best_trial["params"]["n_attention_heads"])
    model.load_state_dict(torch.load(path_model))
    model.to(device)
    model.eval()

    smiles_inputs = tokenizer_smiles(smile, return_tensors="pt", truncation=True, padding=True, max_length=150).to(device)
    with torch.no_grad():
        smiles_outputs = model_smiles(**smiles_inputs)
    attention_mask_smiles = smiles_inputs['attention_mask'].unsqueeze(-1).expand(smiles_outputs.last_hidden_state.size()).float()
    chem_emb = (smiles_outputs.last_hidden_state * attention_mask_smiles).sum(1) / attention_mask_smiles.sum(1).clamp(min=1e-9)
    chem_emb = chem_emb.float()

    fasta_inputs = tokenizer_fasta(fasta, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    with torch.no_grad():
        fasta_outputs = model_fasta(**fasta_inputs)
    attention_mask_fasta = fasta_inputs['attention_mask'].unsqueeze(-1).expand(fasta_outputs.last_hidden_state.size()).float()
    fasta_emb = (fasta_outputs.last_hidden_state * attention_mask_fasta).sum(1) / attention_mask_fasta.sum(1).clamp(min=1e-9)
    fasta_emb = fasta_emb.float()

    print("chem_emb shape:", chem_emb.shape)
    print("fasta_emb shape:", fasta_emb.shape)

    with torch.no_grad():
        output_tensor = model(chem_emb, fasta_emb)
        print("Raw model output:", output_tensor)
        affinity = output_tensor.squeeze().item()
    

    return affinity

if __name__ == "__main__":

    smile = "Nc1ncnc(N2CCC(CC2)c2nc(cn2CCN2CCCC2)-c2ccc(F)c(c2)C(F)(F)F)c1F"
    fasta = "MRRRRRRDGFYPAPDFRDREAEDMAGVFDIDLDQPEDAGSEDELEEGGQLNESMDHGGVGPYELGMEHCEKFEISETSVNRGPEKIRPECFELLRVLGKGGYGKVFQVRKVTGANTGKIFAMKVLKKAMIVRNAKDTAHTKAERNILEEVKHPFIVDLIYAFQTGGKLYLILEYLSGGELFMQLEREGIFMEDTACFYLAEISMALGHLHQKGIIYRDLKPENIMLNHQGHVKLTDFGLCKESIHDGTVTHTFCGTIEYMAPEILMRSGHNRAVDWWSLGALMYDMLTGAPPFTGENRKKTIDKILKCKLNLPPYLTQEARDLLKKLLKRNAASRLGAGPGDAGEVQAHPFFRHINWEELLARKVEPPFKPLLQSEEDVSQFDSKFTRQTPVDSPDDSTLSESANQVFLGFTYVAPSVLESVKEKFSFEPKIRSPRRFIGSPRTPVSPVKFSPGDFWGRGASASTANPQTPVEYPMETSGIEQMDVTMSGEASAPLPIRQPNSGPYKKQAFPMISKRPEHLRMNL"
    path_model = r"models\checkpoints\cnn_affinity\trial_1_loss_0.1974.pth"

    best_trial = get_best_trial("models/cnn_affinity.db", study_name="cnn_affinity")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer_fasta = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D",trust_remote_code=True)
    model_fasta = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D", torch_dtype=torch.float16).to(device)
    tokenizer_smiles = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    model_smiles = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k").to(device)


    raw = predict_affinity(smile, fasta, path_model, best_trial=best_trial)
    affinity = 10 ** -raw # As we used -log10(affinity) during training, we need to reverse this transformation

    affinity = round(affinity, 2)
    print(f"Predicted affinity: {affinity}")
