#ifndef MHA_RESIDUAL_CONNECTION_H
#define MHA_RESIDUAL_CONNECTION_H

#include "FFMatrixBlock.h"
#include "JoinComp.h"

#include "Lambda.h"
#include "LambdaCreationFunctions.h"

#include <cmath>

using namespace pdb;

class MHAResConnection : public JoinComp<FFMatrixBlock,FFMatrixBlock, FFMatrixBlock> {

private:
    uint32_t context_size;
    uint32_t embed_size;

public:
  ENABLE_DEEP_COPY


  MHAResConnection() {
        this->context_size = 10;
        this->embed_size = 64;
    }


  Lambda<bool> getSelection(Handle<FFMatrixBlock> in1,
                            Handle<FFMatrixBlock> in2) override {

    return (makeLambda(in1,
                          [](Handle<FFMatrixBlock> &in1) { return true; }) && 
            makeLambda(in2,
                          [](Handle<FFMatrixBlock> &in2) { return true; }));
    // return makeLambda(
    //     in1, in2, [](Handle<FFMatrixBlock> &in1, Handle<FFMatrixBlock> &in2) {
    //       return in1->getBlockColIndex() == in2->getBlockColIndex() 
    //             && in1->getBlockRowIndex() == in2->getBlockRowIndex();
    //     });
  }

  Lambda<Handle<FFMatrixBlock>>
    getProjection(Handle<FFMatrixBlock> in1,
                Handle<FFMatrixBlock> in2) override {
        return makeLambda(in1, in2,
        [this](Handle<FFMatrixBlock> &in1,
               Handle<FFMatrixBlock> &in2) {

            // load the metadata
            uint32_t inNumRow = in1->getRowNums();
            uint32_t inNumCol = in1->getColNums();
            uint32_t inBlockRowIndex = in1->getBlockRowIndex();
            uint32_t inBlockColIndex = in1->getBlockColIndex();
            // testing purpose
            // std::cout << inNumRow << "," << inNumCol << std::endl;
            // std::cout << inBlockRowIndex << "," << inBlockColIndex << std::endl;

            uint32_t context_size = this->context_size;
            uint32_t embed_size = this->embed_size;

            std::cout << "Model Structure: " << context_size << "," << embed_size << std::endl;
            std::cout << " inNumRow: " << inNumRow << " inNumCol: " << inNumCol << " inBlockRowIndex: " << 
                    inBlockRowIndex << " inBlockColIndex: " << inBlockColIndex << 
                    " inTotalRowNums: " << in1->getTotalRowNums() << " inTotalColNums: " << in1->getTotalColNums() <<   std::endl;
            // std::cout << inBlockRowIndex << "," << inBlockColIndex << std::endl;


            // init weights and bias
            // std::vector<double> embedOutput(sizeEmbed * sizeBatch, 1);
            // std::vector<double> dense0Weight(sizeDense0 * sizeEmbed, 1);
            // std::vector<double> dense0Bias(sizeDense0, 1);
            // std::vector<double> dense0Output(sizeDense0 * sizeBatch, 1);
            // std::vector<double> dense1Weight(sizeDense1 * sizeDense0, 1);
            // std::vector<double> dense1Bias(sizeDense1, 1);
            std::vector<double> sum(context_size * embed_size, 1);
            
            double *inData1 = in1->getValue().rawData->c_ptr();
            double *inData2 = in2->getValue().rawData->c_ptr();
            // dense 0
            // x0 [context_size, embed_size]
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
                x0(inData1, context_size, embed_size);

            // x1 [context_size, embed_size]
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
                x1(inData2, context_size, embed_size);
            
            // y0 [context_size, embed_size]
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
                y0(&sum.data()[0], context_size, embed_size);

            // computation for the dense 0
            y0 = x1 + x0;


            // convert result to FFMatrixBlock
            uint32_t I = in1->getRowNums();
            uint32_t J = in1->getColNums();

            pdb::Handle<FFMatrixBlock> resultFFMatrixBlock =
                pdb::makeObject<FFMatrixBlock>(
                    in1->getBlockRowIndex(), in1->getBlockColIndex(), I, J,
                    in1->getTotalRowNums(), in1->getTotalColNums());

            double *outData = resultFFMatrixBlock->getValue().rawData->c_ptr();

            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
                resultMatrix(outData, context_size, embed_size);

            resultMatrix = y0;
            // testing purpose
            /* std::cout << "===========Product result ==========" << std::endl;
            std::cout << resultMatrix.rows() << "," << resultMatrix.cols()
                      << std::endl;
            std::cout << resultFFMatrixBlock->getRowNums() << ","
                      << resultFFMatrixBlock->getColNums() << std::endl;
            std::cout << y1.rows() << "," << y1.cols() << std::endl; */
            return resultFFMatrixBlock;
        });
    }
};

#endif
