#ifndef MHA_LAYER_NORM_H
#define MHA_LAYER_NORM_H

#include "FFMatrixBlock.h"
#include "SelectionComp.h"
#include "Lambda.h"
#include "LambdaCreationFunctions.h"

#include <cmath>

using namespace pdb;

// enum class SumActivation { Sigmod = 1, Tanh };

class MHALayerNorm : public SelectionComp<FFMatrixBlock, FFMatrixBlock> {

public:
  ENABLE_DEEP_COPY

  MHALayerNorm() {
    }

    // layernorm(uint32_t sizeEmbed, uint32_t sizeDense0, uint32_t sizeDense1) {
    //     this->sizeEmbed = sizeEmbed;
    //     this->sizeDense0 = sizeDense0;
    //     this->sizeDense1 = sizeDense1;
    // }

  Lambda<bool> getSelection(Handle<FFMatrixBlock> checkMe) override {

    return makeLambda(checkMe,
                          [](Handle<FFMatrixBlock> &checkMe) { return true; });
  }

  Lambda<Handle<FFMatrixBlock>>
  getProjection(Handle<FFMatrixBlock> in1) override {
    return makeLambda(
        in1, [this](Handle<FFMatrixBlock> &in1) {
          if (FFMatrixBlock::librayCode == EIGEN_CODE) {
            // get the sizes

            uint32_t I = in1->getRowNums();
            uint32_t J = in1->getColNums();

            pdb::Handle<FFMatrixBlock> resultFFMatrixBlock =
                pdb::makeObject<FFMatrixBlock>(
                    in1->getBlockRowIndex(), in1->getBlockColIndex(), I, J,
                    in1->getTotalRowNums(), in1->getTotalColNums());

            double *outData = resultFFMatrixBlock->getValue().rawData->c_ptr();
            double *in1Data = in1->getValue().rawData->c_ptr();

            double sum = 0;

            double numOfelements = I*J;

            for (int i = 0; i < I * J; i++) {
               sum += in1Data[i];
            }

            double mean = sum / numOfelements;
            double var = 0.0;

            for(int i = 0; i < I * J; i++){
                var += (in1Data[i] - mean) * (in1Data[i] - mean);
            }

            var /= numOfelements;
            double sd = sqrt(var);
            
            for(int i = 0; i < I * J; i++){
                outData[i] = (in1Data[i] - mean) / sd; 
            }

            return resultFFMatrixBlock;
          } else {
            std::cerr << "Wrong librayCode!" << std::endl;
            exit(1);
          }
        });
  }
};

#endif